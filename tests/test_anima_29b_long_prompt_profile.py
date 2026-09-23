"""Architecture-aware long-prompt profile tests."""

from types import SimpleNamespace

from sd_embed.embedding_funcs import _anima_v3_resolve_long_prompt_strategy


def _pipe(depth: int):
    return SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace(num_layers=depth))
    )


def test_auto_uses_native_length_residual_fusion_for_29b() -> None:
    assert _anima_v3_resolve_long_prompt_strategy(_pipe(40), "auto") == "chunk_residual"


def test_auto_preserves_legacy_concat_for_base_anima() -> None:
    assert _anima_v3_resolve_long_prompt_strategy(_pipe(28), "auto") == "chunk_concat"


def test_explicit_long_prompt_strategy_is_preserved() -> None:
    assert _anima_v3_resolve_long_prompt_strategy(_pipe(40), "chunk_concat") == "chunk_concat"
