"""
Benchmark: skip decoder_fc (lm-head) in generate_context.

generate_context builds context embeddings for context_tps by running the
decoder, but only needs dec_embedding — not dec_logits. Previously decoder_fc
(d_model -> vocab projection) was called and its output discarded.

Compares:
  - BEFORE: compute_logits=True in generate_context (wasted projection)
  - AFTER:  compute_logits=False in generate_context (current code)

Also checks training step is unaffected (generate_context is not called during
training, so timing should be identical).
"""
import statistics
import torch
import torch.nn as nn
from perturbgen.Modules.transformer import PerturbGen

GENE_TO_ROWID = {'<cls>': 2, '<mask>': 1, '<pad>': 0, '<eos>': 3}
VOCAB = 25000
D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 6
D_FF = 3072
SEQ = 128
N_WARMUP = 3
N_REPS = 8


def build_model(device):
    model = PerturbGen(
        tgt_vocab_size=VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_FF, max_seq_length=SEQ + 4,
        dropout=0.0, pred_tps=[3], context_tps=[1, 2], n_total_tps=3,
        encoder='Transformer_encoder', context_mode=True,
        condition_dict=None, gene_to_rowid=GENE_TO_ROWID, seed=42,
    )
    return model.to(device)


def make_generate_inputs(batch, device):
    torch.manual_seed(0)
    src = torch.randint(4, VOCAB, (batch, SEQ), device=device)
    src[:, -20:] = 0
    tgt = {}
    for t in [1, 2, 3]:
        ids = torch.randint(4, VOCAB, (batch, SEQ), device=device)
        ids[:, -20:] = 0
        tgt[f'tgt_input_ids_t{t}'] = ids
    return src, tgt


def make_train_inputs(batch, device):
    torch.manual_seed(1)
    src = torch.randint(4, VOCAB, (batch, SEQ), device=device)
    tgt = {f'tgt_input_ids_t{t}': torch.randint(4, VOCAB, (batch, SEQ), device=device)
           for t in [1, 2, 3]}
    return src, tgt


def time_fn(fn, n_warmup, n_reps, device):
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return times


def bench_generate(device):
    print(f"\n{'='*60}")
    print("GENERATION benchmark  (context_tps=[1,2], pred_tps=[3], 18 iters)")
    print(f"vocab={VOCAB}, d_model={D_MODEL}, layers={NUM_LAYERS}, seq={SEQ}")
    print(f"{'='*60}")

    model = build_model(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    results = {}
    for batch in [4, 16, 32]:
        src, tgt = make_generate_inputs(batch, device)

        # AFTER: compute_logits=False in generate_context (current code)
        def reset():
            model._enc_cache_key = None
            model._ctx_cache_key = None

        def run_after():
            reset()
            with torch.no_grad():
                model.generate(src, tgt, iterations=18)

        times_after = time_fn(run_after, N_WARMUP, N_REPS, device)

        # BEFORE: force compute_logits=True in generate_context by monkey-patching
        orig_call_decoder = model.call_decoder

        def call_decoder_force_logits(*args, **kwargs):
            kwargs['compute_logits'] = True
            return orig_call_decoder(*args, **kwargs)

        model.call_decoder = call_decoder_force_logits

        def run_before():
            reset()
            with torch.no_grad():
                model.generate(src, tgt, iterations=18)

        times_before = time_fn(run_before, N_WARMUP, N_REPS, device)
        model.call_decoder = orig_call_decoder

        med_before = statistics.median(times_before)
        med_after = statistics.median(times_after)
        speedup = med_before / med_after
        fmt_b = [f'{t:.3f}' for t in times_before]
        fmt_a = [f'{t:.3f}' for t in times_after]
        print(f"\nbatch={batch}:")
        print(f"  BEFORE (logits in ctx): {med_before:.3f}s  {fmt_b}")
        print(f"  AFTER  (skip logits):   {med_after:.3f}s  {fmt_a}")
        print(f"  Speedup: {speedup:.2f}x  ({(speedup-1)*100:.1f}% faster)")
        results[batch] = (med_before, med_after, speedup)

    del model
    torch.cuda.empty_cache()
    return results


def bench_training(device):
    print(f"\n{'='*60}")
    print("TRAINING benchmark  (generate_context not called during training)")
    print(f"vocab={VOCAB}, d_model={D_MODEL}, layers={NUM_LAYERS}, seq={SEQ}")
    print(f"{'='*60}")

    model = build_model(device).train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)

    results = {}
    for batch in [4, 16]:
        src, tgt = make_train_inputs(batch, device)

        def step():
            opt.zero_grad()
            outputs = model(src_input_id=src, tgt_input_id_dict=tgt, not_masked=False)
            loss = sum(
                loss_fn(out['dec_logits'].reshape(-1, VOCAB), out['labels'].reshape(-1))
                for out in outputs.values()
            )
            loss.backward()
            opt.step()

        times = time_fn(step, N_WARMUP, N_REPS, device)
        med = statistics.median(times)
        fmt = [f'{t:.3f}' for t in times]
        print(f"\nbatch={batch}: {med:.3f}s/step  {fmt}")
        print(f"  (training is unaffected by this change — generate_context not called)")
        results[batch] = med

    del model, opt
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        gen_results = bench_generate(device)
        train_results = bench_training(device)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'batch':<8} {'before (s)':<14} {'after (s)':<14} {'speedup':<10}")
        for batch, (b, a, s) in gen_results.items():
            print(f"{batch:<8} {b:<14.3f} {a:<14.3f} {s:.2f}x")
