"""Shape/wiring smoke test for EmbeddingFinetuneing's ablation_mode branches.

Needs a GPU: CudnnLstm.__init__ calls self.cuda() unconditionally, so the
'none' and 'embedding_as_input' modes cannot be constructed on a login node.
Run on a GPU node: python scripts/smoke_test_embedding_ablations.py

Checks, for each mode:
  - the module constructs,
  - forward() returns [T, B, ny],
  - the parameter set actually matches what the mode claims to test
    (no adapter/decoder in linear_probe, no adapter in embedding_as_input),
  - the tensors the mode says it ignores really do not affect the output.

That last check is the point of the script: a mode that silently kept using
the static attributes would still produce correctly-shaped output, and the
ablation would be quietly meaningless.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'dmg'))

from dmg.models.neural_networks.embedding_finetuneing import EmbeddingFinetuneing  # noqa: E402
from dmg.models.neural_networks.raw_fm_inputs_finetuneing import (  # noqa: E402
    RawFmInputsFinetuneing,
)

T, B, D, NY = 40, 6, 128, 13
FORCINGS = [f'f{i}' for i in range(7)]
ATTRS = [f'a{i}' for i in range(22)]
FM_TS = [f'p{i}' for i in range(5)]
FM_STATIC = [f'q{i}' for i in range(48)]


def make_config(mode, adapter_type='dual_residual'):
    return {
        'nn': {
            'name': 'EmbeddingFinetuneing',
            'forcings': FORCINGS,
            'attributes': ATTRS,
            'embedding_size': D,
            'dropout': 0.1,
            'adapter_type': adapter_type,
            'adapter_params': {
                'dropout': 0.1,
                'combined_dropout': 0.2,
                'hidden_multiplier': 2,
            },
            'use_residual_lstm': True,
            'lstm_hidden_size': 256,
            'ablation_mode': mode,
        }
    }


def make_batch(device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    n_feat = len(FORCINGS) + len(ATTRS)
    xc = torch.randn(T, B, n_feat, generator=g).to(device)
    emb = torch.randn(T, B, D, generator=g).to(device)
    return {'xc_nn_norm': xc, 'xc_pretrained_norm': emb}


def run(mode, adapter_type='dual_residual'):
    device = 'cuda'
    torch.manual_seed(1234)
    model = EmbeddingFinetuneing(make_config(mode, adapter_type), ny=NY).to(device)
    model.eval()

    batch = make_batch(device)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (T, B, NY), f"{mode}: got {tuple(out.shape)}, want {(T, B, NY)}"

    names = {n for n, _ in model.named_parameters()}
    has_adapter = any(n.startswith('adapter.') for n in names)
    has_decoder = any(n.startswith('decoder.') for n in names)
    n_params = sum(p.numel() for p in model.parameters())

    if mode == 'linear_probe':
        assert not has_adapter, "linear_probe must not build an adapter"
        assert not has_decoder, "linear_probe must not build an LSTM decoder"
        expected = D * NY + NY + 2 * D  # Linear + LayerNorm
        assert n_params == expected, f"linear_probe: {n_params} params, want {expected}"
    elif mode == 'embedding_as_input':
        assert not has_adapter, "embedding_as_input must not build an adapter"
        assert has_decoder, "embedding_as_input needs the LSTM decoder"
        assert model.decoder.nx == D + len(FORCINGS), (
            f"decoder nx={model.decoder.nx}, want {D + len(FORCINGS)} "
            "(embedding channels + forcings, statics dropped)"
        )
    else:
        assert has_decoder, "full model needs the LSTM decoder"
        # adapter_type 'none' builds nn.Identity, which contributes no
        # parameters -- so absence of adapter.* params is expected there.
        assert has_adapter == (adapter_type != 'none'), (
            f"adapter_type={adapter_type}: adapter params present={has_adapter}"
        )

    # --- the inputs this mode claims to ignore must not move the output ---
    n_ts = len(FORCINGS)
    perturbed = {k: v.clone() for k, v in batch.items()}
    if mode == 'linear_probe':
        perturbed['xc_nn_norm'] += 100.0            # all task features
        label = 'task forcings+attributes'
    elif mode == 'embedding_as_input':
        perturbed['xc_nn_norm'][..., n_ts:] += 100.0  # statics only
        label = 'static attributes'
    else:
        perturbed = None
        label = None

    if perturbed is not None:
        with torch.no_grad():
            out2 = model(perturbed)
        drift = (out2 - out).abs().max().item()
        assert drift == 0.0, (
            f"{mode}: perturbing {label} changed the output by {drift:.3e}; "
            "they are supposed to be ignored"
        )
        print(f"  {label} confirmed ignored (max drift {drift:.1e})")

    # The embedding must still matter in every mode. Perturb it with random
    # per-channel noise, NOT a constant offset or a global rescale -- every
    # mode runs the embedding through LayerNorm first, which removes both of
    # those exactly, so a constant-shift test would pass on float error alone
    # even if the embedding were disconnected.
    g = torch.Generator(device='cpu').manual_seed(99)
    noise = torch.randn(T, B, D, generator=g).to(device)
    perturbed_emb = {k: v.clone() for k, v in batch.items()}
    perturbed_emb['xc_pretrained_norm'] += noise
    with torch.no_grad():
        out3 = model(perturbed_emb)
    emb_drift = (out3 - out).abs().max().item()
    assert emb_drift > 1e-4, (
        f"{mode}: embedding barely affected the output (max drift "
        f"{emb_drift:.3e}); it may not be wired through"
    )

    print(
        f"PASS {mode:<20} out={tuple(out.shape)} params={n_params:,} "
        f"adapter={has_adapter} decoder={has_decoder} emb_effect={emb_drift:.3e}"
    )


def run_raw_fm_inputs():
    """The no-foundation-model control, which is a separate model class.

    Its 'xc_pretrained_norm' carries RAW variables (5 ts + 48 static), not an
    embedding, so it is checked separately from the ablation_mode branches.
    """
    device = 'cuda'
    torch.manual_seed(1234)
    cfg = {
        'nn': {
            'name': 'RawFmInputsFinetuneing',
            'forcings': FORCINGS,
            'attributes': ATTRS,
            'pretrained_time_series_vars': FM_TS,
            'pretrained_static_vars': FM_STATIC,
            'embedding_size': D,
            'dropout': 0.1,
            'adapter_type': 'dual_residual',
            'adapter_params': {
                'dropout': 0.1,
                'combined_dropout': 0.2,
                'hidden_multiplier': 2,
            },
            'use_residual_lstm': True,
            'lstm_hidden_size': 256,
        }
    }
    model = RawFmInputsFinetuneing(cfg, ny=NY).to(device)
    model.eval()

    g = torch.Generator(device='cpu').manual_seed(7)
    xc = torch.randn(T, B, len(FORCINGS) + len(ATTRS), generator=g).to(device)
    # NnDualLoader broadcasts the static block across time; mimic that exactly,
    # otherwise the t=0 slice the model takes would not be representative.
    fm_ts = torch.randn(T, B, len(FM_TS), generator=g).to(device)
    fm_st = torch.randn(1, B, len(FM_STATIC), generator=g).to(device).expand(T, -1, -1)
    fm = torch.cat([fm_ts, fm_st], dim=-1).contiguous()
    batch = {'xc_nn_norm': xc, 'xc_pretrained_norm': fm}

    with torch.no_grad():
        out = model(batch)
    assert out.shape == (T, B, NY), f"raw_fm_inputs: got {tuple(out.shape)}"

    # The adapter must actually be sized for task+FM inputs, not task alone.
    assert model.adapter.n_time_features == len(FORCINGS) + len(FM_TS)
    assert model.adapter.n_static_features == len(ATTRS) + len(FM_STATIC)

    # Both FM halves must move the output -- if either were silently dropped
    # (e.g. a bad split of xc_pretrained_norm) the control would be vacuous.
    for label, sl in (('FM time vars', slice(0, len(FM_TS))),
                      ('FM static vars', slice(len(FM_TS), None))):
        p = {k: v.clone() for k, v in batch.items()}
        p['xc_pretrained_norm'][..., sl] += torch.randn(
            T, B, p['xc_pretrained_norm'][..., sl].shape[-1], generator=g
        ).to(device)
        with torch.no_grad():
            o = model(p)
        drift = (o - out).abs().max().item()
        assert drift > 1e-4, f"raw_fm_inputs: {label} had no effect ({drift:.3e})"
        print(f"  {label} affect the output (drift {drift:.3e})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PASS {'raw_fm_inputs':<20} out={tuple(out.shape)} params={n_params:,}")


if __name__ == '__main__':
    assert torch.cuda.is_available(), "needs a GPU (CudnnLstm calls .cuda())"
    run('none')
    run('none', adapter_type='none')  # the "no adapter layer" ablation
    run('linear_probe')
    run('embedding_as_input')
    run_raw_fm_inputs()
    print("\nAll ablation modes OK.")
