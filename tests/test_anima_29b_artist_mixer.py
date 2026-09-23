from types import SimpleNamespace

import torch.nn as nn

from sd_embed.anima_artist_mixer_plus_diffusers import (
    ArtistSpec,
    _ANIMA_29B_BASE_TO_EXPANDED,
    _ANIMA_29B_INSERTED_TO_SOURCE,
    _ANIMA_29B_SOURCE_FOR_BLOCK,
    _default_get_blocks,
    build_layer_artist_weights,
)


def test_anima_29b_manifest_mapping_consumes_all_28_source_blocks() -> None:
    assert len(_ANIMA_29B_SOURCE_FOR_BLOCK) == 40
    assert set(_ANIMA_29B_SOURCE_FOR_BLOCK.values()) == set(range(28))
    assert _ANIMA_29B_SOURCE_FOR_BLOCK[2] == 1
    assert _ANIMA_29B_SOURCE_FOR_BLOCK[36] == 24
    assert _ANIMA_29B_SOURCE_FOR_BLOCK[39] == 27


def test_inherited_block_mapping_matches_released_29b_layout() -> None:
    assert len(_ANIMA_29B_BASE_TO_EXPANDED) == 28
    assert len(set(_ANIMA_29B_BASE_TO_EXPANDED)) == 28
    assert set(_ANIMA_29B_BASE_TO_EXPANDED).isdisjoint(_ANIMA_29B_INSERTED_TO_SOURCE)
    assert set(_ANIMA_29B_BASE_TO_EXPANDED) | set(_ANIMA_29B_INSERTED_TO_SOURCE) == set(range(40))


def test_global_artist_weight_targets_inherited_blocks_only() -> None:
    weights = build_layer_artist_weights([ArtistSpec("a", {"global": 1.0})], 40)
    assert len(weights) == 40
    assert all(weights[i][0] == 1.0 for i in _ANIMA_29B_BASE_TO_EXPANDED)
    assert all(weights[i][0] == 0.0 for i in _ANIMA_29B_INSERTED_TO_SOURCE)


def test_component_weight_moves_to_corresponding_inherited_block() -> None:
    specs = [ArtistSpec("a", {"style": 1.0})]
    w28 = build_layer_artist_weights(specs, 28)
    w40 = build_layer_artist_weights(specs, 40)
    assert w40[1][0] == w28[1][0]
    assert w40[35][0] == w28[24][0]
    assert w40[39][0] == w28[27][0]
    assert w40[2][0] == 0.0
    assert w40[36][0] == 0.0


def test_default_get_blocks_supports_diffusers_anima_nested_core() -> None:
    blocks = nn.ModuleList([nn.Identity() for _ in range(40)])
    pipe = SimpleNamespace(transformer=SimpleNamespace(core=SimpleNamespace(transformer_blocks=blocks)))
    assert _default_get_blocks(pipe) is blocks
