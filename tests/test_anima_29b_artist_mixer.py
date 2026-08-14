from types import SimpleNamespace

import torch.nn as nn

from sd_embed.anima_artist_mixer_plus_diffusers import (
    ArtistSpec,
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


def test_global_artist_weight_reaches_all_40_blocks() -> None:
    weights = build_layer_artist_weights([ArtistSpec("a", {"global": 1.0})], 40)
    assert len(weights) == 40
    assert all(weights[i][0] == 1.0 for i in range(40))


def test_inserted_block_inherits_source_component_weight() -> None:
    specs = [ArtistSpec("a", {"style": 1.0})]
    w28 = build_layer_artist_weights(specs, 28)
    w40 = build_layer_artist_weights(specs, 40)
    assert w40[2][0] == w28[1][0]
    assert w40[36][0] == w28[24][0]
    assert w40[39][0] == w28[27][0]


def test_default_get_blocks_supports_diffusers_anima_nested_core() -> None:
    blocks = nn.ModuleList([nn.Identity() for _ in range(40)])
    pipe = SimpleNamespace(transformer=SimpleNamespace(core=SimpleNamespace(transformer_blocks=blocks)))
    assert _default_get_blocks(pipe) is blocks
