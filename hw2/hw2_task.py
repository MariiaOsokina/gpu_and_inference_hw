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
# The optimized loop — the 5 fixes
# optimized_loop() + generate_optimized() stack these (cumulative):
#
#   Fix              What it does                                       Speedup
#   ---------------  ------------------------------------------------  --------
#   #1 KV cache      Feed only the new token each step instead of      1.00 ->
#                    the whole sequence. The one algorithmic win.       4.79x
#   #2 defer .item() Collect tokens as tensors, convert to ints once   4.79 ->
#                    at the end. 1 CPU<->GPU sync instead of 128.       4.79x
#   #3 bf16          REVERTED. GPU got faster but HF Llama re-casts     4.79 ->
#                    fp32 buffers every step, adding CPU overhead       4.32x
#                    that lost net.
#   #4 inference_    Turn off autograd bookkeeping entirely            4.79 ->
#      mode          (stronger than eval()).                            6.61x
#   #5 torch.compile Fuse the many tiny per-step ops into a few        6.61 ->
#                    Triton kernels.                                   10.71x
#   #6 TF32          Let fp32 matmuls use tensor cores without          10.71 ->
#                    changing dtype (so no per-step casts like bf16).   11.20x
#
#   Final optimized loop: 11.20x over the V0 slow baseline.
#
# Biggest win: #1 KV cache. It is the only fix that changes the loop's
#   algorithm (O(prompt x steps) -> O(steps)). Because this model is tiny the
#   loop is CPU-launch-bound — the CPU spends ~3x longer launching kernels than
#   the GPU spends running them — so the KV cache both delivers the largest jump
#   and exposes that bottleneck. #4 and #5, which cut per-launch/per-op CPU
#   overhead, are the next-biggest wins; #3 (bf16) backfired because it sped up
#   the GPU (not the bottleneck) while adding CPU work (the bottleneck).
#
# ----------------------------------------------------------------------------
# Setup: 1x NVIDIA L40S. V0 runs in ~1.64s here (the README's ~21s is other
# hardware), but grading is the ratio and both loops run on the same GPU, so
# the tiers still apply. Speedups are from the un-profiled time_generation()
# run; "Profiler" numbers below are CPU/CUDA self-time over the 12-step profile.
#
# Trace note (v1_optimized_trace.json): profile() runs before the timed run, so
# the trace captures torch.compile's one-time compile/autotune storm
# (InductorBenchmarker, thousands of autotuning aten::fill_/zero_), which dwarfs
# the real loop. The steady-state loop is the "Torch-Compiled Region: 0/2"
# graph (~1.1ms/decode step). The timed run is unaffected — it runs warm graphs.
#
# ----------------------------------------------------------------------------
# Profiler / timing evidence per fix:
#
#   #1 KV cache: V0 re-feeds the whole growing sequence every step
#     (O(prompt_len * n_steps)); with use_cache the model returns a DynamicCache
#     and each step feeds just the 1 new token. aten::mm CUDA time 108ms -> 16ms
#     (-85%) over 12 steps, kernels switched SGEMM -> GEMV (the bandwidth-bound,
#     memory-bound decode signature).
#
#   #2 defer .item(): V0 syncs CPU<->GPU every step via .item(); instead I keep
#     (1,1) tensors in a list and convert once with torch.cat(...).tolist(). The
#     per-step sync disappears from the trace, but wall-clock is flat at this
#     scale (per-step GPU work is only ~1.8ms). Kept — structurally correct and
#     pays off on larger models.
#
#   #3 bf16 (REVERTED): GPU side worked (Tensor Core + flash-attention kernels,
#     CUDA time -66%), but HF Llama re-casts fp32 buffers (RoPE inv_freq,
#     scales) every forward: +269 aten::to, CPU time 82ms -> 192ms. Since the
#     loop is launch-bound, that CPU cost outweighed the GPU win (clean A/B:
#     fp32 6.61x vs bf16 5.95x). bf16 should win on a larger model where per-
#     step GPU work dominates.
#
#   #4 inference_mode: eval() only disables dropout; autograd still builds graphs
#     and bumps version counters on the hundreds of in-place ops/step.
#     inference_mode skips all of it. Per-op saving is below profiler resolution
#     (op table barely moves) but removes ~100ms of CPU bookkeeping across 128
#     steps -> 0.25s.
#
#   #5 torch.compile: the profiles showed CPU self-time (~73ms) ~3x CUDA (~22ms)
#     over 12 steps — launch-bound. compile traces the forward into fused Triton
#     kernels (e.g. the whole RMSNorm+residual chain becomes one kernel),
#     cutting launches and per-op overhead. Dynamo emits 3 graphs (prefill,
#     first-decode, steady-state decode) with no per-step recompile; it compiles
#     on the first profile() calls, so the timed run is warm.
#
#   #6 TF32 (set_float32_matmul_precision("high")): routes fp32 matmuls through
#     the tensor-core TF32 path. Unlike bf16 it does NOT change dtype, so no
#     per-step casts (the regression that sank #3). Set inside
#     generate_optimized() so the V0 baseline stays full-fp32. Matmuls switched
#     to cutlass_80_tensorop_s1688gemm (TF32); small gain because compile already
#     removed launch overhead, leaving little matmul time in the ~1.1ms/step.
