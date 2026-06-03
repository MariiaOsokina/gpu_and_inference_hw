import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.tensor(times).median())


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    total_flops = num_elements * num_ops * 2  # 2 FLOPs per iteration (multiply + add)

    if variant == "compiled":
        # fused kernel: one read + one write at the kernel boundary
        total_bytes = num_elements * bytes_per_element * 2
    else:
        # eager: each iteration materializes two intermediates (mul result, add result)
        # per iteration: read acc + read x (mul) + write tmp + read tmp + read x (add) + write acc
        # simplified: 4 tensors touched per iteration × num_ops, plus initial read + final write
        total_bytes = num_elements * bytes_per_element * (4 * num_ops + 2)

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#Q1 — #These compiled element-wise ops are memory-bound — the slow part is moving the 256 MB tensor in and out of VRAM, not the math. 
# Because the kernel is fused, it reads the tensor once and writes it once regardless of num_ops, so the bytes moved (and therefore the time) 
# stay fixed at ~0.868 ms every step.
#  Performance = FLOPs ÷ time. Each step doubles the FLOPs while time stays constant, so achieved TFLOP/s doubles too. 
# The extra math is ~10× faster than the memory transfer and overlaps with it — the cores compute on data already 
# loaded while the rest streams in — so the math "hides" under the fixed transfer time and the clock doesn't move.
#   All points stay left of the ridge point (106 FLOP/Byte on the L40S): same bytes moved, 
# just more useful work per pass, so they climb the slanted bandwidth ceiling instead of adding runtime. 
# This would only stop once the math exceeds the transfer time (crossing into compute-bound) — 
# which never happens here, 
# since even 64 ops only reaches AI = 16.


# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.


#Q2 — Peak FLOP/s requires saturating the GPU — keeping all its SMs (worker crews; an H100 has 132) busy. 
# A 1024×1024 matmul produces only ~1M outputs; split across so many SMs, each crew gets a tiny slice, finishes fast, then sits idle. 
# Two effects follow:

# 1 - Under-saturation: too little work per crew means launch/setup overhead becomes a large fraction of the total, 
# so only ~25% of peak compute is used.
# 2 - cuBLAS needs large tiles: the matmul library splits matrices into square tiles, one per crew, 
# and its optimizations only run at full speed when the matrices are big enough to form full tiles. 
# A small 1024 matrix can't fill them.

#Meanwhile the 128 ops element-wise op runs over 67M elements — far more independent work — 
# so it fully saturates the GPU and posts high FLOP/s.

# Hence the upset: the matmul is a compute-heavy type of work but there's too little of it to fill a big GPU, 
# while the element-wise op is simpler but has tons of work. 
# Making the matmul bigger (2048, 4096) gives enough work to saturate the chip, 
# and it pulls ahead toward the compute ceiling.

#(Note: on my L40S this flip didn't actually occur — matmul-1024 hit 23.5 TFLOP/s vs 19.8 for 128-ops. 
# The effect is described for the larger H100, where idle SMs are easier to create.)


# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#Q3 — A runtime that finally starts climbing means the bottleneck has switched from
# moving data to doing math — the kernel has crossed the ridge point from
# memory-bound into compute-bound territory.
#
# Mechanism: for small ops the math "hid" underneath the fixed memory transfer (the
# cores compute on already-loaded data while the rest streams in), so adding math was
# free and the clock stayed flat. That only works while the math is shorter than the
# transfer. Once you pile on enough operations, the math itself grows longer than the
# memory transfer — it no longer fits underneath, sticks out past the end, and the
# runtime starts to rise. At that point the limiting resource is the FP32 arithmetic
# units (the FMA/ALU hardware), not memory bandwidth.
#
# Where the ridge sits decides when this happens, and the two GPUs differ sharply:
#                 ridge point   64 ops (AI=16)   128 ops (AI=32)
#   H100          20            memory-bound     compute-bound
#   My L40S       106           memory-bound     memory-bound
# On an H100 (ridge=20) the crossover lands exactly in the 64->128 gap: 64 ops is
# still left of the ridge, 128 ops is past it, so runtime jumps between them.
# On my L40S (ridge=106) both 64 (AI=16) and 128 (AI=32) are still far left of the
# ridge, so both stay memory-bound and my measured runtime is flat (~0.868 ms both).
# The jump the question describes doesn't occur on my hardware.
# (Why the different ridges: ridge = peak compute / peak bandwidth. The H100's huge
# 3,350 GB/s bandwidth feeds its cores easily, so it becomes math-limited sooner —
# low ridge. The L40S's 864 GB/s keeps it data-starved much longer — high ridge.)

# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#Q4 — The difference comes down to how many times the data travels to and from VRAM.

#Compiled = fused into one kernel. 
# All the math is done in a single pass: read the tensor once, 
# do every operation while the data sits in fast on-chip registers, write once — no matter how many ops. 
# So bytes moved stay fixed, and arithmetic intensity climbs as you add math (0.25 → 32 FLOP/Byte). 
# The points march rightward and upward along the bandwidth ceiling.

#Eager = a separate kernel for every step. 
# Each * and each + runs on its own, writing its result back to VRAM and re-reading it for the next step. 
# So the bytes moved grow right along with num_ops — more math always drags more memory traffic with it. 
# Arithmetic intensity barely moves (0.083 → 0.125 FLOP/Byte), the points stay pinned in the bottom-left, 
# and runtime scales almost linearly (1 op ≈ 2 ms → 128 ops ≈ 329 ms).

#Compiled moves the data once and reuses it; eager re-loads it for every operation, 
# so it's permanently memory-bound and gets dramatically slower as ops increase.

