"""
Benchmark: PerturbGen.generate() wall-clock timing.

Measures median time for model.generate() under realistic settings
(Transformer_encoder, context_tps=[1,2], pred_tps=[3], 18 MaskGIT iterations).

Usage:
    # Record baseline (before refactor):
    python perturbgen/tests/benchmark_generation_efficiency.py

    # After refactor — print speedup vs baseline:
    python perturbgen/tests/benchmark_generation_efficiency.py --baseline <seconds>
"""

import argparse
import statistics
import time

import torch

from perturbgen.Modules.transformer import PerturbGen

GENE_TO_ROWID = {'<cls>': 2, '<mask>': 1, '<pad>': 0, '<eos>': 3}


def build_model(d_model=128, num_heads=4, num_layers=4, seq=256, vocab=500,
                context_tps=None, pred_tps=None, n_total_tps=3,
                context_mode=True, device='cpu', seed=42, compile_model=False):
    if pred_tps is None:
        pred_tps = [3]
    model = PerturbGen(
        tgt_vocab_size=vocab,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 2,
        max_seq_length=seq + 4,
        dropout=0.0,
        pred_tps=pred_tps,
        n_total_tps=n_total_tps,
        encoder='Transformer_encoder',
        context_mode=context_mode,
        context_tps=context_tps,
        condition_dict=None,
        gene_to_rowid=GENE_TO_ROWID,
        seed=seed,
        compile_model=compile_model,
    )
    model.eval()
    return model.to(device)


def make_inputs(batch, seq, vocab, device, seed=0):
    torch.manual_seed(seed)
    src = torch.randint(4, vocab, (batch, seq)).to(device)
    src[:, -10:] = 0
    tgt = {}
    for t in [1, 2, 3]:
        ids = torch.randint(4, vocab, (batch, seq)).to(device)
        ids[:, -10:] = 0
        tgt[f'tgt_input_ids_t{t}'] = ids
    return src, tgt


def time_generate(model, src, tgt, iterations, device, n_warmup=1, n_reps=5,
                  disable_ctx_cache=False):
    def reset_caches():
        model._enc_cache_key = None
        model._ctx_cache_key = None

    def run():
        if disable_ctx_cache:
            # Defeat context cache by clearing it after every forward call so it
            # never hits — isolates the cost of recomputing context each iteration
            orig_forward = model.forward
            call_count = [0]
            def patched_forward(*a, **kw):
                out = orig_forward(*a, **kw)
                model._ctx_cache_key = None
                return out
            model.forward = patched_forward
            with torch.no_grad():
                model.generate(src, tgt, iterations=iterations)
            model.forward = orig_forward
        else:
            with torch.no_grad():
                model.generate(src, tgt, iterations=iterations)

    for _ in range(n_warmup):
        reset_caches()
        run()
        if device != 'cpu':
            torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        reset_caches()
        if device != 'cpu':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
    return times


def report(label, times, baseline=None, n_reps=5):
    med = statistics.median(times)
    fmt = [f'{t:.3f}' for t in times]
    print(f"\n{'='*60}")
    print(f"Results [{label}] (median of {n_reps} runs):")
    print(f"  Time:  {med:.3f}s   {fmt}")
    if baseline is not None:
        speedup = baseline / med
        print(f"  Speedup vs baseline: {speedup:.2f}x  "
              f"({(speedup - 1) * 100:.1f}% {'faster' if speedup > 1 else 'slower'})")
    print(f"{'='*60}")
    return med


def run_config(label, device, batch, seq, vocab, d_model, num_layers,
               iterations, context_tps, pred_tps, n_reps, baseline=None,
               ctx_cache_ablation=False, compile_model=False):
    ctx_str = str(context_tps) if context_tps else 'None'
    print(f"\n[{label}] device={device}  batch={batch}  seq={seq}  vocab={vocab}"
          f"  d_model={d_model}  layers={num_layers}  compile={compile_model}")
    print(f"  context_tps={ctx_str}  pred_tps={pred_tps}  iterations={iterations}")
    model = build_model(
        d_model=d_model, num_heads=4, num_layers=num_layers,
        seq=seq, vocab=vocab,
        context_tps=context_tps, pred_tps=pred_tps,
        context_mode=(context_tps is not None),
        device=device,
        compile_model=compile_model,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    src, tgt = make_inputs(batch, seq, vocab, device)

    if ctx_cache_ablation and context_tps is not None:
        # Run WITH and WITHOUT context cache to isolate its contribution
        times_cached = time_generate(model, src, tgt, iterations, device,
                                     n_reps=n_reps, disable_ctx_cache=False)
        times_nocache = time_generate(model, src, tgt, iterations, device,
                                      n_reps=n_reps, disable_ctx_cache=True)
        med_cached = statistics.median(times_cached)
        med_nocache = statistics.median(times_nocache)
        speedup = med_nocache / med_cached
        fmt_c = [f'{t:.3f}' for t in times_cached]
        fmt_n = [f'{t:.3f}' for t in times_nocache]
        print(f"\n{'='*60}")
        print(f"Context cache ablation [{label}] (median of {n_reps} runs):")
        print(f"  WITH cache:    {med_cached:.3f}s   {fmt_c}")
        print(f"  WITHOUT cache: {med_nocache:.3f}s   {fmt_n}")
        print(f"  Context cache speedup: {speedup:.2f}x  "
              f"({(speedup - 1) * 100:.1f}% faster)")
        print(f"  (context_tps={ctx_str}, {len(context_tps)} ctx decoder runs "
              f"× {iterations} iters = {len(context_tps) * (iterations - 1)} "
              f"calls saved per generate)")
        print(f"{'='*60}")
        return med_cached

    times = time_generate(model, src, tgt, iterations, device, n_reps=n_reps)
    med = report(label, times, baseline=baseline, n_reps=n_reps)
    return med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=float, default=None,
                        help='Baseline median seconds to compare against')
    parser.add_argument('--reps', type=int, default=5)
    args = parser.parse_args()

    baseline = args.baseline
    n_reps = args.reps

    # --- Config 1: context cache ablation on CPU ---
    med_ctx = run_config(
        label='CPU | context_tps=[1,2] | pred_tps=[3]',
        device='cpu',
        batch=4, seq=200, vocab=300,
        d_model=128, num_layers=4,
        iterations=18,
        context_tps=[1, 2], pred_tps=[3],
        n_reps=n_reps,
        baseline=baseline,
        ctx_cache_ablation=True,
    )

    # --- Config 2: no context (control — cache has no effect here) ---
    run_config(
        label='CPU | no-context | pred_tps=[1]',
        device='cpu',
        batch=4, seq=200, vocab=300,
        d_model=128, num_layers=4,
        iterations=18,
        context_tps=None, pred_tps=[1],
        n_reps=n_reps,
        baseline=None,
    )

    # --- GPU configs (if available) ---
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n[GPU] {torch.cuda.get_device_name(device)}")

        # Context cache ablation on GPU
        run_config(
            label='GPU | context_tps=[1,2] | pred_tps=[3]',
            device=device,
            batch=16, seq=512, vocab=500,
            d_model=256, num_layers=6,
            iterations=18,
            context_tps=[1, 2], pred_tps=[3],
            n_reps=n_reps,
            baseline=None,
            ctx_cache_ablation=True,
        )

        run_config(
            label='GPU | no-context | pred_tps=[1]',
            device=device,
            batch=16, seq=512, vocab=500,
            d_model=256, num_layers=6,
            iterations=18,
            context_tps=None, pred_tps=[1],
            n_reps=n_reps,
            baseline=None,
        )

        # --- torch.compile ablation on GPU ---
        print(f"\n{'='*60}")
        print("torch.compile ablation (encoder + decoder blocks)")
        print(f"{'='*60}")
        for batch_size in [16, 64, 128]:
            compile_kwargs = dict(
                device=device, batch=batch_size, seq=512, vocab=500,
                d_model=256, num_layers=6, iterations=18,
                context_tps=[1, 2], pred_tps=[3], n_reps=n_reps,
            )
            med_no_compile = run_config(
                label=f'GPU batch={batch_size} | no compile', **compile_kwargs)
            med_compiled = run_config(
                label=f'GPU batch={batch_size} | compile_model=True',
                compile_model=True, **compile_kwargs)
            if med_no_compile and med_compiled:
                speedup = med_no_compile / med_compiled
                print(f"\n  [batch={batch_size}] torch.compile speedup: {speedup:.2f}x  "
                      f"({(speedup - 1) * 100:.1f}% {'faster' if speedup > 1 else 'slower'})")
    else:
        print('\n[GPU] CUDA not available — skipping GPU configs.')

    # Summary hint
    print('\nTo compare before/after:')
    print(f'  python perturbgen/tests/benchmark_generation_efficiency.py --baseline {med_ctx:.3f}')


if __name__ == '__main__':
    main()
