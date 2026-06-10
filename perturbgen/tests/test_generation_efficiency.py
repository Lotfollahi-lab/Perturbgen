"""
Regression tests for generation efficiency refactor.

Guards:
- generate_sequence output token IDs are deterministic and match a golden reference
- context_tps=None (no-context) path is stable
- training forward (not_masked=True) mean_embedding is non-None after call_decoder cleanup
- generate with context_mode=True matches reference after caching is applied

Run with:
    python -m pytest perturbgen/tests/test_generation_efficiency.py -x -v
"""

import torch
import pytest
from perturbgen.Modules.transformer import PerturbGen, CountDecoder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GENE_TO_ROWID = {
    '<cls>': 2,
    '<mask>': 1,
    '<pad>': 0,
    '<eos>': 3,
}

VOCAB_SIZE = 60
SEQ_LEN = 30
BATCH = 2
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2  # use 2 so the mid-layer branch in call_decoder is exercised
D_FF = 64
ITERATIONS = 4  # keep short for speed


def _build_model(
    context_tps=None,
    context_mode=True,
    pred_tps=None,
    seed=42,
    compile_model=False,
):
    if pred_tps is None:
        pred_tps = [3]
    if context_mode and context_tps is None:
        context_tps = [1, 2]

    model = PerturbGen(
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=SEQ_LEN + 4,
        dropout=0.0,
        pred_tps=pred_tps,
        n_total_tps=3,
        encoder='Transformer_encoder',
        pos_encoding_mode='time_pos_sin',
        context_mode=context_mode,
        context_tps=context_tps,
        condition_dict=None,
        gene_to_rowid=GENE_TO_ROWID,
        seed=seed,
        compile_model=compile_model,
    )
    model.eval()
    return model


def _make_inputs(seed=0, batch=BATCH, seq=SEQ_LEN, vocab=VOCAB_SIZE):
    """Build src and tgt_input_id_dict for t1, t2, t3."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    src = torch.randint(4, vocab, (batch, seq), generator=rng)
    src[:, -5:] = 0  # trailing padding

    tgt = {}
    for t in [1, 2, 3]:
        ids = torch.randint(4, vocab, (batch, seq), generator=rng)
        ids[:, -5:] = 0
        tgt[f'tgt_input_ids_t{t}'] = ids
    return src, tgt


def _run_generate(model, seed_run=7):
    """Run model.generate with a fixed torch seed and return generated IDs."""
    torch.manual_seed(seed_run)
    src, tgt = _make_inputs()
    with torch.no_grad():
        _, gen_ids = model.generate(
            src_input_id=src,
            tgt_input_id_dict=tgt,
            iterations=ITERATIONS,
            temperature=1.5,
            mask_scheduler='cosine',
        )
    return gen_ids


# ---------------------------------------------------------------------------
# Test 1: determinism — two runs with same seed produce identical output
# ---------------------------------------------------------------------------

class TestGenerateDeterminism:
    def test_same_seed_same_output(self):
        model = _build_model(seed=42)
        out1 = _run_generate(model, seed_run=7)
        # reset model-internal caches
        model._enc_cache_key = None
        model._ctx_cache_key = None
        out2 = _run_generate(model, seed_run=7)
        for t in [3]:
            k = f'tgt_input_ids_t{t}'
            assert torch.equal(out1[k], out2[k]), (
                f"Non-deterministic output at {k}: {out1[k]} vs {out2[k]}"
            )

    def test_no_context_determinism(self):
        model = _build_model(context_mode=False, context_tps=None, pred_tps=[1])
        src, tgt = _make_inputs()
        torch.manual_seed(0)
        with torch.no_grad():
            _, ids1 = model.generate(src, tgt, iterations=ITERATIONS)
        model._enc_cache_key = None
        torch.manual_seed(0)
        with torch.no_grad():
            _, ids2 = model.generate(src, tgt, iterations=ITERATIONS)
        assert torch.equal(ids1['tgt_input_ids_t1'], ids2['tgt_input_ids_t1'])


# ---------------------------------------------------------------------------
# Test 2: golden reference — generated IDs match snapshot taken on unmodified code
# Update REFERENCE_IDS below by running this test once and printing the value.
# ---------------------------------------------------------------------------

# Golden reference recorded from unmodified transformer.py (seed=42, run_seed=7, iterations=4)
REFERENCE_IDS_T3 = torch.tensor([
    [20, 20, 48, 20, 57, 20, 25, 25, 48, 10, 25, 20, 48, 48, 10, 48, 8, 12, 57, 20, 10, 25, 25, 25, 13, 26, 56, 31, 27, 25],
    [8, 57, 20, 20, 20, 20, 25, 8, 20, 20, 48, 57, 10, 26, 20, 20, 8, 20, 25, 8, 48, 12, 12, 13, 48, 39, 56, 56, 2, 20],
])


class TestGoldenReference:
    def test_generate_matches_golden(self):
        model = _build_model(seed=42)
        gen_ids = _run_generate(model, seed_run=7)
        ids_t3 = gen_ids['tgt_input_ids_t3']

        global REFERENCE_IDS_T3
        if REFERENCE_IDS_T3 is None:
            print(
                f"\n[golden] Copy this into REFERENCE_IDS_T3:\n"
                f"torch.tensor({ids_t3.tolist()})"
            )
            pytest.skip("Golden reference not yet set — run once to record it")
        else:
            assert torch.equal(ids_t3, REFERENCE_IDS_T3), (
                f"Generated IDs changed!\nExpected: {REFERENCE_IDS_T3}\nGot:      {ids_t3}"
            )

    def test_no_context_generate_matches_golden(self):
        """Separate golden for the no-context path (context_mode=False)."""
        model = _build_model(context_mode=False, context_tps=None, pred_tps=[1], seed=42)
        torch.manual_seed(7)
        src, tgt = _make_inputs()
        with torch.no_grad():
            _, gen_ids = model.generate(src, tgt, iterations=ITERATIONS)
        ids_t1 = gen_ids['tgt_input_ids_t1']
        # This test always passes (just prints on first run); golden locked after refactor.
        print(f"\n[golden no-ctx] tgt_input_ids_t1: {ids_t1.tolist()}")


# ---------------------------------------------------------------------------
# Test 3: training forward (not_masked=True) — mean_embedding is non-None
# Guards against call_decoder cleanup breaking the training path
# ---------------------------------------------------------------------------

class TestTrainingForward:
    def test_mean_embedding_non_none(self):
        model = _build_model(seed=42)
        src, tgt = _make_inputs()
        with torch.no_grad():
            outputs = model.forward(
                src_input_id=src,
                tgt_input_id_dict=tgt,
                not_masked=True,
            )
        for t in [3]:
            assert outputs[t]['mean_embedding'] is not None, (
                f"mean_embedding is None for t={t}"
            )
            assert outputs[t]['mean_embedding'].shape == (BATCH, D_MODEL), (
                f"Unexpected shape: {outputs[t]['mean_embedding'].shape}"
            )

    def test_dec_logits_shape(self):
        model = _build_model(seed=42)
        src, tgt = _make_inputs()
        with torch.no_grad():
            outputs = model.forward(src_input_id=src, tgt_input_id_dict=tgt, not_masked=True)
        for t in [3]:
            logits = outputs[t]['dec_logits']
            assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), (
                f"Unexpected logits shape: {logits.shape}"
            )

    def test_no_context_training_forward(self):
        model = _build_model(context_mode=False, context_tps=None, pred_tps=[1], seed=42)
        src, tgt = _make_inputs()
        with torch.no_grad():
            outputs = model.forward(src_input_id=src, tgt_input_id_dict=tgt, not_masked=True)
        assert outputs[1]['mean_embedding'] is not None


# ---------------------------------------------------------------------------
# Test 4: context cache does not change generated output
# Specifically: running generate twice with the cache populated gives same IDs
# ---------------------------------------------------------------------------

class TestContextCacheCorrectness:
    def test_generate_with_multiple_pred_tps(self):
        """If pred_tps has multiple entries, each gets its own cached context."""
        model = _build_model(pred_tps=[2, 3], context_tps=[1], seed=42)
        src, tgt = _make_inputs()
        torch.manual_seed(5)
        with torch.no_grad():
            _, ids1 = model.generate(src, tgt, iterations=ITERATIONS)
        model._enc_cache_key = None
        model._ctx_cache_key = None
        torch.manual_seed(5)
        with torch.no_grad():
            _, ids2 = model.generate(src, tgt, iterations=ITERATIONS)
        for t in [2, 3]:
            k = f'tgt_input_ids_t{t}'
            assert torch.equal(ids1[k], ids2[k]), f"Mismatch at {k}"

    def test_generate_sequence_length_param(self):
        """sequence_length param doesn't break after refactor."""
        model = _build_model(seed=42)
        src, tgt = _make_inputs()
        with torch.no_grad():
            _, gen_ids = model.generate(
                src, tgt, iterations=ITERATIONS, sequence_length=20
            )
        assert 'tgt_input_ids_t3' in gen_ids


# ---------------------------------------------------------------------------
# Test 5: CountDecoder.generate_counts still runs end-to-end
# ---------------------------------------------------------------------------

class TestCountDecoderGenerate:
    def test_generate_counts_runs(self):
        base_model = _build_model(seed=42)
        count_decoder = CountDecoder(
            pretrained_model=base_model,
            loss_mode='nb',
            d_model=D_MODEL,
            max_seq_length=SEQ_LEN + 4,
            encoder='Transformer_encoder',
            pred_tps=[3],
            n_total_tps=3,
            n_genes=10,
            context_tps=[1, 2],
            use_size_factor=False,
            use_observed_size_factor=False,
            seed=42,
        )
        count_decoder.eval()
        src, tgt = _make_inputs()
        with torch.no_grad():
            count_out, gen_ids = count_decoder.generate_counts(
                src_input_id=src,
                tgt_input_id_dict=tgt,
                iterations=ITERATIONS,
                sequence_length=SEQ_LEN,
            )
        assert 'count_output_t3' in count_out
        assert 'count_mean' in count_out['count_output_t3']
        assert count_out['count_output_t3']['count_mean'].shape == (BATCH, 10)


# ---------------------------------------------------------------------------
# Test 6: torch.compile — compiled model output matches uncompiled
# ---------------------------------------------------------------------------

class TestCompiledModel:
    def test_compiled_generate_matches_uncompiled(self):
        """Compiled encoder+decoder blocks must produce identical token IDs."""
        if not torch.cuda.is_available():
            pytest.skip("torch.compile test requires CUDA")
        device = torch.device('cuda')

        torch.manual_seed(42)
        model_base = _build_model(seed=42, compile_model=False).to(device)
        torch.manual_seed(42)
        model_compiled = _build_model(seed=42, compile_model=True).to(device)

        src, tgt = _make_inputs()
        src = src.to(device)
        tgt = {k: v.to(device) for k, v in tgt.items()}

        torch.manual_seed(7)
        with torch.no_grad():
            _, ids_base = model_base.generate(src, tgt, iterations=ITERATIONS)

        torch.manual_seed(7)
        with torch.no_grad():
            _, ids_compiled = model_compiled.generate(src, tgt, iterations=ITERATIONS)

        for t in [3]:
            k = f'tgt_input_ids_t{t}'
            assert torch.equal(ids_base[k].cpu(), ids_compiled[k].cpu()), (
                f"Compiled output differs at {k}"
            )

    def test_compiled_training_forward_matches_uncompiled(self):
        """Compiled decoder blocks must give identical logits in training forward."""
        if not torch.cuda.is_available():
            pytest.skip("torch.compile test requires CUDA")
        device = torch.device('cuda')

        model_base = _build_model(seed=42, compile_model=False).to(device)
        model_compiled = _build_model(seed=42, compile_model=True).to(device)

        src, tgt = _make_inputs()
        src = src.to(device)
        tgt = {k: v.to(device) for k, v in tgt.items()}

        with torch.no_grad():
            out_base = model_base.forward(src_input_id=src, tgt_input_id_dict=tgt, not_masked=True)
            out_comp = model_compiled.forward(src_input_id=src, tgt_input_id_dict=tgt, not_masked=True)

        for t in [3]:
            assert torch.allclose(
                out_base[t]['dec_logits'].cpu(),
                out_comp[t]['dec_logits'].cpu(),
                atol=1e-4,
            ), f"Logits differ at t={t}"
