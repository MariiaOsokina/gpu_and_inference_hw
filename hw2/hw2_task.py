import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # Fix #1: KV cache (use_cache=True + past_key_values).
    # Fix #2: defer .item() to end of loop. The slow loop calls .item() every
    # step, forcing a host sync (CPU waits for all queued GPU work, copies one
    # int back, resumes). Instead, store next_token_id as a tensor in a Python
    # list during the loop, then concatenate and convert to ints once at the
    # end — one sync total instead of n_steps.
    # Fix #4: torch.inference_mode() — disables autograd graph construction and
    # version-counter bookkeeping. model.eval() already disables dropout but
    # leaves autograd active; inference_mode is the stronger inference-only
    # context that skips per-op gradient tracking entirely.
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = [next_token_id]

        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token_id)

        return torch.cat(generated_tokens, dim=1).squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    # Fix #3 (reverted): bf16 weights/KV. On the GPU side bf16 worked — matmuls
    # dispatched to Tensor Core kernels and CUDA self-time fell ~66% — but HF
    # Llama casts several fp32 buffers (RoPE inv_freq, scales) every forward
    # pass, adding hundreds of per-step aten::to / cudaFuncSetAttribute calls.
    # At this model scale per-step GPU work is already sub-millisecond, so the
    # extra CPU overhead outweighed the GPU savings (4.79x -> 4.32x). Reverted
    # to fp32; see writeup. fp32 + inference_mode measured 6.61x.

    # Fix #6: enable TF32. Lets fp32 matmuls use the Ada/Ampere tensor-core
    # TF32 path (10-bit mantissa, fp32 range) instead of the slow full-fp32
    # kernels. Unlike bf16 (fix #3), this does NOT change the model dtype, so
    # it adds no per-step cast ops — the regression that sank bf16. Set here
    # (not at module level) so it applies only to the optimized run; the V0
    # slow baseline already ran and stays at default full-fp32 precision.
    torch.set_float32_matmul_precision("high")

    model = build_model(torch.float32)

    # Fix #5: torch.compile. The profile shows this loop is CPU-launch-bound
    # (CPU self-time ~3x the CUDA self-time over 12 steps): hundreds of tiny
    # ops per step, each paying Python dispatch + a separate cudaLaunchKernel.
    # torch.compile traces the forward into a fused graph, cutting the number
    # of launches and the per-op overhead — directly attacking the bottleneck.
    # Compilation happens on the first calls (during profile()), so the timed
    # run below executes the already-warm graphs.
    model = torch.compile(model)

    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "Optimized")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Hardware: 1x NVIDIA L40S (compute capability 8.9). All numbers are the
# Speedup that time_generation() reports for the optimized loop against the
# unmodified V0 slow baseline (128 tokens from a 1024-token prompt). The V0
# baseline measures ~1.63-1.64s across runs. "Profiler" figures are CPU/CUDA
# self-time totals over the 12-step profile run, used to explain *why* a fix
# helped (or did not); the Speedup column is from the un-profiled timed run.
#
# Note on the baseline: the README example baseline is ~21s, but on the L40S
# this tiny 2-layer model runs the V0 loop in ~1.64s. The grading metric is
# the *ratio*, and both sides run on the same GPU, so the tiers still apply.
# A side effect of the small absolute time is that the loop becomes CPU-launch
# -bound quickly, which shapes the results below.
#
# Note on the optimized trace (v1_optimized_trace.json): because torch.compile
# (#5) compiles and autotunes on the first calls, and profile() runs first,
# the optimized trace captures the one-time compilation/autotuning storm
# (InductorBenchmarker, CachingAutotuner, thousands of aten::fill_/zero_ from
# autotuning scratch buffers, _recursive_joint_graph_passes). That bookkeeping
# dwarfs the actual generation in the trace. The part that reflects the real
# steady-state loop is the compiled decode graph, labeled "Torch-Compiled
# Region: 0/2" (runs once per decode step, ~1.1ms each); Dynamo also emits two
# one-shot graphs for prefill and the first-decode shape. The un-profiled
# time_generation() run (the reported Speedup) is unaffected since it executes
# the already-warm graphs.
#
# ----------------------------------------------------------------------------
# Per-fix progression (each fix is cumulative on top of the previous, except
# bf16 which was measured and then reverted):
#
#   V0  slow baseline ........................................ 1.00x  (1.64s)
#   #1  + KV cache ........................................... 4.79x  (0.34s)
#   #2  + defer .item() ...................................... 4.79x  (0.34s)
#   #3  + bf16 (REVERTED) .................................... 4.32x  (0.38s)
#   #4  + torch.inference_mode() ............................. 6.61x  (0.25s)
#   #5  + torch.compile() ................................... 10.71x  (0.15s)
#   #6  + TF32 matmul precision ............................. 11.20x  (0.15s)
#
#   Final optimized loop: 11.20x over V0.
#
# ----------------------------------------------------------------------------
# What each fix changed and why:
#
#   Fix #1 — KV cache (use_cache=True, past_key_values).
#     1.00x -> 4.79x.
#     The slow loop re-feeds the whole growing sequence every step, recomputing
#     K,V for every prompt position it already processed. Total work scales
#     O(prompt_len * n_steps). With use_cache=True the model returns a
#     DynamicCache; each decode step feeds only the 1 new token and the cache
#     appends one K,V column. Profiler: aten::mm self-CUDA fell 108ms -> 16ms
#     (-85%) over 12 steps, and the matmul kernels switched from SGEMM to GEMV
#     (matrix-vector, gemv2T_kernel) because each decode step is shape (1,1,H).
#     The decode GEMV kernels are bandwidth-bound (read full weight matrix x a
#     1-column input) — the classic memory-bound decode signature.
#
#   Fix #2 — Defer .item() to end of loop.
#     4.79x -> 4.79x (no wall-clock change).
#     The slow loop calls .item() every step, forcing a host sync (CPU waits
#     for all queued GPU work, copies one int back, resumes). I keep tokens as
#     (1,1) tensors in a Python list and convert once at the end with a single
#     torch.cat(...).tolist() — one sync instead of 128. Profiler confirms the
#     CPU self-time dropped ~2.4ms over 12 steps and the per-step
#     cudaMemcpy/sync events disappeared from inside the loop. Wall-clock did
#     NOT move: per-step GPU work is only ~1.8ms and the CPU was not queueing
#     far enough ahead for sync removal to matter at this scale. Kept anyway —
#     it is structurally correct and would pay off on a larger model.
#
#   Fix #3 — bf16 weights and KV cache (REVERTED).
#     4.79x -> 4.32x (regressed).
#     The GPU side worked exactly as intended: kernels switched to bf16 Tensor
#     Core variants (cutlass_80_tensorop_bf16_*, and attention switched to
#     flash attention), and CUDA self-time fell 22ms -> 7.6ms (-66%) over 12
#     steps. But HF Llama keeps several fp32 buffers (RoPE inv_freq, attention
#     scales) and casts them on every forward pass: the profiler showed +269
#     aten::to and +173 aten::_to_copy calls plus 372 cudaFuncSetAttribute
#     configurations for the Tensor Core kernels. CPU self-time jumped 82ms ->
#     192ms (+135%). Because the loop is CPU-launch-bound at this scale, the
#     extra CPU overhead outweighed the GPU savings. Confirmed by a clean A/B
#     with everything else fixed: fp32 = 6.61x vs bf16 = 5.95x. Reverted to
#     fp32. (bf16 would likely win on a larger model where per-step GPU work
#     dominates, or once the casts are fused away.)
#
#   Fix #4 — torch.inference_mode() around the loop.
#     4.79x -> 6.61x.
#     model.eval() (set in build_model) only disables dropout/BN; autograd is
#     still active, so every forward builds a graph, saves activations, and
#     bumps version counters on in-place ops. inference_mode is the stronger
#     inference-only context that skips all of that. The per-op saving is tiny
#     (below profiler resolution, so the summary table barely moves), but HF
#     Llama runs hundreds of in-place ops per step, so across 128 steps it
#     removed ~100ms of pure CPU bookkeeping -> 0.25s. This is a good example
#     of a fix that is invisible in the op table but clear in wall-clock.
#
#   Fix #5 — torch.compile(model).
#     6.61x -> 10.71x.
#     The diagnosis from the profiles was that the loop is CPU-launch-bound:
#     CPU self-time (~73ms) was ~3x the CUDA self-time (~22ms) over 12 steps,
#     i.e. hundreds of tiny ops per step each paying Python dispatch + a
#     separate cudaLaunchKernel. torch.compile traces the forward into fused
#     Triton kernels (e.g. one kernel named
#     triton_red_fused_add_embedding_mean_mul_pow_rsqrt collapses the whole
#     RMSNorm+residual chain), cutting the number of launches and per-op
#     overhead — directly attacking the bottleneck. Dynamo produced 3 graphs
#     (prefill, a transitional first-decode shape, and the steady-state decode
#     graph that runs every step) with no per-step recompilation. Compilation
#     happens on the first calls inside profile(), so the timed run is warm.
#
#   Fix #6 — TF32 matmul precision (set_float32_matmul_precision("high")).
#     10.71x -> 11.20x.
#     Lets fp32 matmuls use the L40S tensor-core TF32 path (10-bit mantissa,
#     full fp32 range) instead of slow full-fp32 kernels. Unlike bf16 (#3),
#     this does NOT change the model dtype, so it adds no per-step cast ops —
#     the regression that sank bf16. Set inside generate_optimized() (not at
#     module level) so it applies only to the optimized run; the V0 baseline
#     already ran and stays at default full-fp32 precision. Profiler: the
#     compiled matmuls switched to cutlass_80_tensorop_s1688gemm (TF32) kernels
#     and the prior TF32 warning disappeared. Small wall-clock gain because,
#     after compile removed the launch overhead, matmuls were already a small
#     fraction of the remaining ~1.1ms/step — little matmul time left to save.
#
# ----------------------------------------------------------------------------
# Biggest impact: Fix #1, the KV cache (1.00x -> 4.79x).
#   It is the only fix that changes the algorithmic cost of the loop. The slow
#   loop's total work is O(prompt_len * n_steps) because every decode step
#   re-prefills the entire context. The KV cache drops the dominant Q/K/V
#   projection and MLP work to O(n_steps) (attention still reads the growing
#   cache, O(prompt_len + n_steps) per step). Every other fix is a
#   constant-factor improvement on top of that; #1 changes the asymptotics, so
#   it both delivers the single largest jump and is what makes the later fixes
#   (which target per-step overhead) worthwhile — there are no longer giant
#   re-prefill matmuls hiding the launch overhead that #4 and #5 remove.
#
# Second-biggest: Fix #5, torch.compile (6.61x -> 10.71x). After the KV cache
#   exposed that the loop was CPU-launch-bound, fusing the per-step ops into a
#   few Triton kernels was the highest-leverage remaining change.
