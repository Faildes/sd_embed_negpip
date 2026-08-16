import torch

from sd_embed.embedding_funcs import (
    _ANIMA_CONDITIONING_MAX_LENGTH,
    _anima_v3_fuse_long_prompt_conditions,
    _anima_v3_normalize_weight_energy,
)


def test_weight_energy_normalization_keeps_unity_exact():
    factors = torch.ones(2, 8, 1)
    mask = torch.ones(2, 8, 1)
    out = _anima_v3_normalize_weight_energy(factors, mask)
    assert torch.equal(out, factors)


def test_weight_energy_normalization_preserves_relative_emphasis_and_unit_rms():
    factors = torch.tensor([[[1.0], [1.0], [1.5], [0.8]]], dtype=torch.float32)
    mask = torch.ones_like(factors)
    out = _anima_v3_normalize_weight_energy(factors, mask)
    rms = out.square().mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6)
    assert torch.allclose(out[:, 2] / out[:, 0], torch.tensor([[1.5]]), atol=1e-6)


def test_public_chunk_concat_is_bounded_to_native_anima_length():
    a = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8)
    b = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8)
    out = _anima_v3_fuse_long_prompt_conditions(
        [a, b], strength=1.0, chunk_decay=1.0, mode="chunk_concat", anchor_tokens=0
    )
    assert out.shape == a.shape


def test_raw_concat_keeps_legacy_unbounded_behavior():
    a = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8)
    b = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8)
    out = _anima_v3_fuse_long_prompt_conditions(
        [a, b], strength=1.0, chunk_decay=1.0, mode="raw_concat", anchor_tokens=0
    )
    assert out.shape[1] == _ANIMA_CONDITIONING_MAX_LENGTH * 2


def test_safe_chunk_concat_does_not_mix_later_layout_windows():
    base = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8)
    other = torch.randn(1, _ANIMA_CONDITIONING_MAX_LENGTH, 8) * 10.0 + 20.0
    out = _anima_v3_fuse_long_prompt_conditions(
        [base, other], strength=1.0, chunk_decay=1.0, mode="chunk_concat", anchor_tokens=0
    )
    assert torch.equal(out, base)
