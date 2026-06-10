"""
Benchmark: PerturbGen training forward+backward at production dimensions.

Compares wall-clock time per training step with and without torch.compile,
at production scale: vocab=25000, d_model=768, seq=2048, batch=64.

Usage:
    python perturbgen/tests/benchmark_training_compile.py
"""

import statistics
import time

import torch
import torch.nn as nn

from perturbgen.Modules.transformer import PerturbGen

GENE_TO_ROWID = {'<cls>': 2, '<mask>': 1, '<pad>': 0, '<eos>': 3}

VOCAB = 25000
SEQ = 128
D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 6
D_FF = 3072
BATCH = 32
N_WARMUP = 2
N_REPS = 5
PRED_TPS = [3]
CONTEXT_TPS = [1, 2]
N_TOTAL_TPS = 3


def build_model(compile_model: bool, device: str) -> PerturbGen:
    model = PerturbGen(
        tgt_vocab_size=VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=SEQ + 4,
        dropout=0.1,
        pred_tps=PRED_TPS,
        context_tps=CONTEXT_TPS,
        n_total_tps=N_TOTAL_TPS,
        encoder='Transformer_encoder',
        context_mode=True,
        condition_dict=None,
        gene_to_rowid=GENE_TO_ROWID,
        seed=42,
        compile_model=compile_model,
    )
    return model.to(device).train()


def make_batch(device: str, seed: int = 0):
    torch.manual_seed(seed)
    src = torch.randint(4, VOCAB, (BATCH, SEQ), device=device)
    src[:, -50:] = 0
    tgt = {}
    for t in [1, 2, 3]:
        ids = torch.randint(4, VOCAB, (BATCH, SEQ), device=device)
        ids[:, -50:] = 0
        tgt[f'tgt_input_ids_t{t}'] = ids
    return src, tgt


def time_training_step(model, src, tgt, optimizer, loss_fn, device,
                        n_warmup=N_WARMUP, n_reps=N_REPS):
    def step():
        optimizer.zero_grad()
        outputs = model(
            src_input_id=src,
            tgt_input_id_dict=tgt,
            not_masked=False,
        )
        total_loss = torch.tensor(0.0, device=device)
        for t, out in outputs.items():
            logits = out['dec_logits'].contiguous().view(-1, VOCAB)
            labels = out['labels'].contiguous().view(-1)
            total_loss = total_loss + loss_fn(logits, labels)
        total_loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        step()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return times


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU.")
        return

    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Config: vocab={VOCAB}, d_model={D_MODEL}, seq={SEQ}, batch={BATCH}")
    print(f"        layers={NUM_LAYERS}, heads={NUM_HEADS}, d_ff={D_FF}")
    print(f"        context_tps={CONTEXT_TPS}, pred_tps={PRED_TPS}")
    print(f"Warmup={N_WARMUP}, reps={N_REPS}\n")

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    src, tgt = make_batch(device)

    results = {}
    for compile_flag in [False, True]:
        label = 'compile_model=True' if compile_flag else 'no compile'
        print(f"Building model [{label}]...")
        model = build_model(compile_model=compile_flag, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print(f"  Running {N_WARMUP} warmup + {N_REPS} timed steps...")
        times = time_training_step(model, src, tgt, optimizer, loss_fn, device)
        med = statistics.median(times)
        fmt = [f'{t:.3f}' for t in times]
        print(f"\n{'='*60}")
        print(f"Training step [{label}] (median of {N_REPS} runs):")
        print(f"  Time:  {med:.3f}s   {fmt}")
        print(f"{'='*60}\n")
        results[label] = med

        del model, optimizer
        torch.cuda.empty_cache()

    if len(results) == 2:
        no_compile = results['no compile']
        compiled = results['compile_model=True']
        speedup = no_compile / compiled
        print(f"\n{'='*60}")
        print(f"torch.compile training speedup: {speedup:.2f}x  "
              f"({(speedup - 1) * 100:.1f}% {'faster' if speedup > 1 else 'slower'})")
        print(f"  no compile: {no_compile:.3f}s/step")
        print(f"  compiled:   {compiled:.3f}s/step")
        print(f"  savings:    {(no_compile - compiled) * 1000:.0f}ms/step")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
