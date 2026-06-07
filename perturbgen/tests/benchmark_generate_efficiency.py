"""
Benchmark: generate_sequence hot-loop efficiency
Compares old (4 × .clone() per iteration) vs new (view-based) implementation.

Settings: 100 sequences, seq_len=1000, vocab_size=5000, iterations=18 (default MaskGIT).
This isolates the inner loop so results are reproducible on CPU without a real model.
"""

import time
import sys
import torch
import statistics

sys.path.insert(0, '/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb')
from perturbgen.src.utils import top_k, gumbel_sample


def noise_schedule_stub(ratio, total_tokens, **_):
    """Minimal cosine noise schedule (same as production code)."""
    return torch.cos(ratio * (torch.pi / 2)) * torch.ones_like(total_tokens, dtype=torch.float)


def run_old_loop(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    cond_length: int,
    iterations: int,
    topk_filter_thres: float = 0.9,
    starting_temperature: float = 1.5,
    mask_token: int = 1,
):
    """
    Pre-refactor hot loop: 4 clones per iteration.
      1. tmp_ids[:, cond_length:].clone()
      2. scores[:, cond_length:].clone()
      3. ids_to_keep[:, cond_length:].clone()
      4. logits.clone() before top_k
    """
    torch.manual_seed(42)
    max_neg_value = -torch.finfo(torch.float32).max

    tmp_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    tmp_ids[:, :cond_length] = 0
    tmp_ids[torch.rand_like(tmp_ids.float()) < 0.3] = mask_token
    scores = torch.zeros(batch_size, seq_len)
    scores[:, :cond_length] = max_neg_value
    total_tokens = torch.full((batch_size,), seq_len - cond_length)

    iteration_ratios = torch.linspace(0, 1, iterations)
    for iteration, steps_until_x0 in zip(iteration_ratios, reversed(range(iterations))):
        rand_mask_prob = noise_schedule_stub(iteration, total_tokens)
        scores.masked_fill_(scores == max_neg_value, max_neg_value)
        unmasked = (scores != max_neg_value).sum(dim=1)
        num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if num_tokens_to_mask.max() > 0:
            indices_to_mask = torch.topk(scores, max(1, num_tokens_to_mask.max()), dim=-1).indices
            for i in range(batch_size):
                mask[i, indices_to_mask[i, :num_tokens_to_mask[i]]] = True
        tmp_ids.masked_fill_(mask, mask_token)
        ids_to_keep = torch.where(mask, torch.zeros_like(tmp_ids), tmp_ids)

        # simulate forward pass output
        logits = torch.randn(batch_size, seq_len - cond_length, vocab_size)

        # OLD: 4 clones
        tmp_ids_ = tmp_ids[:, cond_length:].clone()           # clone 1
        scores_ = scores[:, cond_length:].clone()             # clone 2
        ids_to_keep_ = ids_to_keep[:, cond_length:].clone()  # clone 3

        idx = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, idx, max_neg_value)
        filtered_logits = top_k(logits.clone(), topk_filter_thres)  # clone 4

        temperature = starting_temperature * (steps_until_x0 / iterations)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])
        scores_.masked_fill_(~is_mask, max_neg_value)
        scores[:, cond_length:] = scores_
        tmp_ids[:, cond_length:] = tmp_ids_

    return tmp_ids


def run_new_loop(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    cond_length: int,
    iterations: int,
    topk_filter_thres: float = 0.9,
    starting_temperature: float = 1.5,
    mask_token: int = 1,
):
    """
    Post-refactor hot loop: 0 clones — views only.
    """
    torch.manual_seed(42)
    max_neg_value = -torch.finfo(torch.float32).max

    tmp_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    tmp_ids[:, :cond_length] = 0
    tmp_ids[torch.rand_like(tmp_ids.float()) < 0.3] = mask_token
    scores = torch.zeros(batch_size, seq_len)
    scores[:, :cond_length] = max_neg_value
    total_tokens = torch.full((batch_size,), seq_len - cond_length)

    iteration_ratios = torch.linspace(0, 1, iterations)
    for iteration, steps_until_x0 in zip(iteration_ratios, reversed(range(iterations))):
        rand_mask_prob = noise_schedule_stub(iteration, total_tokens)
        scores.masked_fill_(scores == max_neg_value, max_neg_value)
        unmasked = (scores != max_neg_value).sum(dim=1)
        num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if num_tokens_to_mask.max() > 0:
            indices_to_mask = torch.topk(scores, max(1, num_tokens_to_mask.max()), dim=-1).indices
            for i in range(batch_size):
                mask[i, indices_to_mask[i, :num_tokens_to_mask[i]]] = True
        tmp_ids.masked_fill_(mask, mask_token)
        ids_to_keep = torch.where(mask, torch.zeros_like(tmp_ids), tmp_ids)

        logits = torch.randn(batch_size, seq_len - cond_length, vocab_size)

        # NEW: views, no clones
        tmp_ids_ = tmp_ids[:, cond_length:]
        ids_to_keep_ = ids_to_keep[:, cond_length:]

        idx = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, idx, max_neg_value)
        filtered_logits = top_k(logits, topk_filter_thres)  # no clone

        temperature = starting_temperature * (steps_until_x0 / iterations)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])
        scores_.masked_fill_(~is_mask, max_neg_value)
        scores[:, cond_length:] = scores_
        tmp_ids[:, cond_length:] = tmp_ids_

    return tmp_ids


def benchmark_cpu(fn, n_warmup=2, n_repeats=5, **kwargs):
    for _ in range(n_warmup):
        fn(**kwargs)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(**kwargs)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_gpu(fn, device, n_warmup=3, n_repeats=10, **kwargs):
    """Use CUDA events for accurate GPU timing (accounts for async execution)."""
    kwargs = {**kwargs, 'device': device}
    for _ in range(n_warmup):
        fn(**kwargs)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(n_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**kwargs)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / 1000.0)  # ms → s
    return times


def run_old_loop_device(batch_size, seq_len, vocab_size, cond_length, iterations,
                        topk_filter_thres=0.9, starting_temperature=1.5,
                        mask_token=1, device='cpu'):
    """Old loop with device support for GPU benchmarking."""
    torch.manual_seed(42)
    max_neg_value = -torch.finfo(torch.float32).max
    tmp_ids = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    tmp_ids[:, :cond_length] = 0
    tmp_ids[torch.rand_like(tmp_ids.float()) < 0.3] = mask_token
    scores = torch.zeros(batch_size, seq_len, device=device)
    scores[:, :cond_length] = max_neg_value
    total_tokens = torch.full((batch_size,), seq_len - cond_length, device=device)

    iteration_ratios = torch.linspace(0, 1, iterations)
    for iteration, steps_until_x0 in zip(iteration_ratios, reversed(range(iterations))):
        rand_mask_prob = noise_schedule_stub(iteration, total_tokens)
        scores.masked_fill_(scores == max_neg_value, max_neg_value)
        unmasked = (scores != max_neg_value).sum(dim=1)
        num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if num_tokens_to_mask.max() > 0:
            indices_to_mask = torch.topk(scores, max(1, num_tokens_to_mask.max().item()), dim=-1).indices
            for i in range(batch_size):
                mask[i, indices_to_mask[i, :num_tokens_to_mask[i]]] = True
        tmp_ids.masked_fill_(mask, mask_token)
        ids_to_keep = torch.where(mask, torch.zeros_like(tmp_ids), tmp_ids)
        logits = torch.randn(batch_size, seq_len - cond_length, vocab_size, device=device)

        tmp_ids_   = tmp_ids[:, cond_length:].clone()
        scores_    = scores[:, cond_length:].clone()
        ids_to_keep_ = ids_to_keep[:, cond_length:].clone()
        idx = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, idx, max_neg_value)
        filtered_logits = top_k(logits.clone(), topk_filter_thres)
        temperature = starting_temperature * (steps_until_x0 / iterations)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])
        scores_.masked_fill_(~is_mask, max_neg_value)
        scores[:, cond_length:] = scores_
        tmp_ids[:, cond_length:] = tmp_ids_
    return tmp_ids


def run_new_loop_device(batch_size, seq_len, vocab_size, cond_length, iterations,
                        topk_filter_thres=0.9, starting_temperature=1.5,
                        mask_token=1, device='cpu'):
    """New loop with device support for GPU benchmarking."""
    torch.manual_seed(42)
    max_neg_value = -torch.finfo(torch.float32).max
    tmp_ids = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    tmp_ids[:, :cond_length] = 0
    tmp_ids[torch.rand_like(tmp_ids.float()) < 0.3] = mask_token
    scores = torch.zeros(batch_size, seq_len, device=device)
    scores[:, :cond_length] = max_neg_value
    total_tokens = torch.full((batch_size,), seq_len - cond_length, device=device)

    iteration_ratios = torch.linspace(0, 1, iterations)
    for iteration, steps_until_x0 in zip(iteration_ratios, reversed(range(iterations))):
        rand_mask_prob = noise_schedule_stub(iteration, total_tokens)
        scores.masked_fill_(scores == max_neg_value, max_neg_value)
        unmasked = (scores != max_neg_value).sum(dim=1)
        num_tokens_to_mask = (unmasked.float() * rand_mask_prob).long()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if num_tokens_to_mask.max() > 0:
            indices_to_mask = torch.topk(scores, max(1, num_tokens_to_mask.max().item()), dim=-1).indices
            for i in range(batch_size):
                mask[i, indices_to_mask[i, :num_tokens_to_mask[i]]] = True
        tmp_ids.masked_fill_(mask, mask_token)
        ids_to_keep = torch.where(mask, torch.zeros_like(tmp_ids), tmp_ids)
        logits = torch.randn(batch_size, seq_len - cond_length, vocab_size, device=device)

        tmp_ids_     = tmp_ids[:, cond_length:]
        ids_to_keep_ = ids_to_keep[:, cond_length:]
        idx = ids_to_keep_.unsqueeze(1).expand(-1, seq_len - cond_length, -1)
        logits.scatter_(2, idx, max_neg_value)
        filtered_logits = top_k(logits, topk_filter_thres)
        temperature = starting_temperature * (steps_until_x0 / iterations)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        is_mask = tmp_ids_ == mask_token
        tmp_ids_ = torch.where(is_mask, pred_ids, tmp_ids_)
        probs = logits.softmax(dim=-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        scores_ = 1 - torch.abs(top2[:, :, 0] - top2[:, :, 1])
        scores_.masked_fill_(~is_mask, max_neg_value)
        scores[:, cond_length:] = scores_
        tmp_ids[:, cond_length:] = tmp_ids_
    return tmp_ids


def print_results(label, old_times, new_times, batch_size, seq_len, n_repeats):
    old_med = statistics.median(old_times)
    new_med = statistics.median(new_times)
    speedup = old_med / new_med
    per_clone_mb = (batch_size * seq_len * 4) / 1e6
    print(f"\n{'='*60}")
    print(f"Results [{label}] (median of {n_repeats} runs):")
    print(f"  OLD  {old_med:.3f}s   {[f'{t:.3f}' for t in old_times]}")
    print(f"  NEW  {new_med:.3f}s   {[f'{t:.3f}' for t in new_times]}")
    print(f"  Speedup: {speedup:.2f}x  ({(speedup-1)*100:.1f}% {'faster' if speedup>1 else 'slower'})")
    print(f"  Memory avoided: {per_clone_mb:.1f} MB/clone × 4 × {ITERATIONS} iters"
          f" = {per_clone_mb * 4 * ITERATIONS:.1f} MB per generate() call")
    print(f"{'='*60}")


if __name__ == '__main__':
    BATCH_SIZE  = 20
    SEQ_LEN     = 1000
    VOCAB_SIZE  = 5000
    COND_LENGTH = 0
    ITERATIONS  = 18
    N_WARMUP    = 3
    N_REPEATS   = 10

    print(f"\n{'='*60}")
    print(f"Benchmark: generate_sequence hot loop")
    print(f"  batch={BATCH_SIZE}  seq_len={SEQ_LEN}  vocab={VOCAB_SIZE}")
    print(f"  iterations={ITERATIONS}  clones_removed=4")
    print(f"{'='*60}")

    kwargs = dict(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                  cond_length=COND_LENGTH, iterations=ITERATIONS)

    # ---- CPU ----
    print("\n[CPU] Running OLD loop...")
    old_cpu = benchmark_cpu(run_old_loop, n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs)
    print("[CPU] Running NEW loop...")
    new_cpu = benchmark_cpu(run_new_loop, n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs)
    print_results("CPU", old_cpu, new_cpu, BATCH_SIZE, SEQ_LEN, N_REPEATS)

    # ---- GPU ----
    if not torch.cuda.is_available():
        print("\n[GPU] CUDA not available — skipping GPU benchmark.")
    else:
        device = torch.device('cuda')
        print(f"\n[GPU] Device: {torch.cuda.get_device_name(device)}")
        print("[GPU] Running OLD loop...")
        old_gpu = benchmark_gpu(run_old_loop_device, device,
                                n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs)
        print("[GPU] Running NEW loop...")
        new_gpu = benchmark_gpu(run_new_loop_device, device,
                                n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs)
        print_results("GPU", old_gpu, new_gpu, BATCH_SIZE, SEQ_LEN, N_REPEATS)

        # also benchmark at production batch size (64)
        print(f"\n[GPU @ batch=64] Running OLD loop...")
        kwargs64 = {**kwargs, 'batch_size': 64}
        old_gpu64 = benchmark_gpu(run_old_loop_device, device,
                                  n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs64)
        print(f"[GPU @ batch=64] Running NEW loop...")
        new_gpu64 = benchmark_gpu(run_new_loop_device, device,
                                  n_warmup=N_WARMUP, n_repeats=N_REPEATS, **kwargs64)
        print_results("GPU batch=64", old_gpu64, new_gpu64, 64, SEQ_LEN, N_REPEATS)

    # ---- correctness ----
    out_old = run_old_loop(**kwargs)
    out_new = run_new_loop(**kwargs)
    print(f"\nOutput match (CPU old == new): {torch.equal(out_old, out_new)}")
