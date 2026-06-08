"""
Microbenchmark: fused vs non-fused Adam optimizer step.

Uses a small model (fits in <8 GB) but production vocab=25000 and d_model=768
so the parameter count and matmul shapes are representative.
Measures only the optimizer .step() time, not the full forward/backward.
"""
import statistics
import time
import torch
import torch.nn as nn
from perturbgen.Modules.transformer import PerturbGen

GENE_TO_ROWID = {'<cls>': 2, '<mask>': 1, '<pad>': 0, '<eos>': 3}
VOCAB = 25000
SEQ = 64
D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 2        # small to fit in MIG slice
D_FF = 3072
BATCH = 2
N_WARMUP = 3
N_REPS = 10


def build_model(device):
    model = PerturbGen(
        tgt_vocab_size=VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_FF, max_seq_length=SEQ + 4,
        dropout=0.1, pred_tps=[3], context_tps=[1, 2], n_total_tps=3,
        encoder='Transformer_encoder', context_mode=True,
        condition_dict=None, gene_to_rowid=GENE_TO_ROWID, seed=42,
    )
    return model.to(device).train()


def run(device):
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Config: vocab={VOCAB}, d_model={D_MODEL}, layers={NUM_LAYERS}, "
          f"seq={SEQ}, batch={BATCH}")

    torch.manual_seed(0)
    src = torch.randint(4, VOCAB, (BATCH, SEQ), device=device)
    tgt = {f'tgt_input_ids_t{t}': torch.randint(4, VOCAB, (BATCH, SEQ), device=device)
           for t in [1, 2, 3]}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    results = {}
    for fused in [False, True]:
        label = f'fused={fused}'
        model = build_model(device)
        n_params = sum(p.numel() for p in model.parameters())
        try:
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=fused and torch.cuda.is_available())
        except (RuntimeError, ValueError, TypeError):
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            label += ' (fallback)'
        print(f"\nBuilding [{label}]  params={n_params:,}")

        # full step to populate gradients for fair optimizer timing
        def full_step():
            opt.zero_grad()
            outputs = model(src_input_id=src, tgt_input_id_dict=tgt, not_masked=False)
            loss = sum(
                loss_fn(out['dec_logits'].reshape(-1, VOCAB), out['labels'].reshape(-1))
                for out in outputs.values()
            )
            loss.backward()
            opt.step()

        # warmup
        for _ in range(N_WARMUP):
            full_step()
            torch.cuda.synchronize()

        # time full step
        times = []
        for _ in range(N_REPS):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            full_step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)

        med = statistics.median(times)
        fmt = [f'{t:.3f}' for t in times]
        print(f"  Step time (median): {med:.3f}s  {fmt}")
        results[label] = med
        del model, opt
        torch.cuda.empty_cache()

    labels = list(results.keys())
    if len(labels) == 2:
        speedup = results[labels[0]] / results[labels[1]]
        print(f"\n{'='*60}")
        print(f"Fused Adam speedup: {speedup:.2f}x  "
              f"({(speedup-1)*100:.1f}% {'faster' if speedup > 1 else 'slower'})")
        print(f"  non-fused: {results[labels[0]]:.3f}s/step")
        print(f"  fused:     {results[labels[1]]:.3f}s/step")
        print(f"{'='*60}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        run('cuda')
