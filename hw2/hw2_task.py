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
    # to fp32; see writeup. (bf16 would likely win on a larger model or once
    # the per-op casts are fused away by torch.compile.)
    model = build_model(torch.float32)
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
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#

# ============================================================================
# Writeup
# ============================================================================
#
# Per-fix speedups (baseline: V0 slow loop, 1.64s for 128 tokens on L40S):
#
#   Fix #1 — KV cache (use_cache=True, past_key_values).
#     1.64s → 0.34s = 4.79x
#     The slow loop re-feeds the entire growing context every step, recomputing
#     K,V for every prompt position. With use_cache=True the model returns a
#     DynamicCache; decode steps feed only the new token and the cache appends
#     one K,V column. Per-step work drops from O(seq_len) to O(1) for the Q/K/V
#     projections and MLP. Profiler confirms: aten::mm self CUDA fell from
#     108 ms to 16 ms (-85%) and the kernels switched from SGEMM to GEMV
#     (matrix-vector) because each decode step is shape (1, 1, hidden).
#
#   Fix #2 — Defer .item() to end of loop.
#     4.79x → 4.79x  (no wall-clock change)
#     Removed 128 per-step host syncs. Profiler shows CPU drop of -2.4 ms over
#     12 profile steps, confirming syncs disappeared. Wall-clock unchanged
#     because per-step GPU work is only ~1.8 ms and the CPU wasn't queueing
#     far enough ahead for sync removal to matter. Pattern still kept; would
#     pay off on a larger model.
#
#   Fix #3 — bf16 weights and KV cache (REVERTED).
#     4.79x → 4.32x  (regressed)
#     GPU side worked: kernels switched to bf16 Tensor Core variants
#     (cutlass_80_tensorop_bf16_*, flash attention), CUDA self time fell from
#     22 ms to 7.6 ms over 12 profile steps (-66%). But HF Llama keeps fp32
#     buffers (RoPE inv_freq, scales) and casts them per forward pass:
#     +269 aten::to and +173 aten::_to_copy calls in 12 steps, plus 372
#     cudaFuncSetAttribute configurations for Tensor Core kernels. CPU self
#     time jumped from 82 ms to 192 ms (+135%). At this model scale per-step
#     GPU work was already ~1.8 ms; the extra CPU overhead outweighed the
#     GPU savings. Reverted.
#
# Biggest impact: KV cache (fix #1).
#   Why: it's the only fix that changes algorithmic cost. The slow loop's
#   total work scales O(prompt_len * n_steps) because every decode step
#   re-prefills the full context. KV cache drops this to
#   O(prompt_len + n_steps * (1 + avg_attn_read)) — essentially O(n_steps)
#   for the dominant Q,K,V projection and MLP work. All other fixes are
#   constant-factor; this one changes the asymptotic structure of the loop.
