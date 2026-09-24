"""Artist contexts should be converted once per runtime shape and dtype."""

import torch
from torch import nn

from sd_embed.anima_artist_mixer_plus_diffusers import DiffusersCrossAttnMixerWrapper


def test_artist_context_cache_reused_across_blocks_and_invalidated_by_dtype():
    shared_state = {
        "artist_contexts": [torch.ones((1, 3, 4))],
        "_runtime_context_cache": {},
    }
    first = DiffusersCrossAttnMixerWrapper(nn.Identity(), shared_state, 0)
    second = DiffusersCrossAttnMixerWrapper(nn.Identity(), shared_state, 1)
    half_context = torch.empty((2, 3, 4), dtype=torch.float16)
    with torch.inference_mode():
        a = first._artist_contexts_for(half_context)[0]
        b = second._artist_contexts_for(half_context)[0]
        assert a is b
        assert a.shape == (2, 3, 4)
        assert a.dtype == torch.float16
        full = first._artist_contexts_for(half_context.float())[0]
        assert full.dtype == torch.float32
        assert full is not a
    assert len(shared_state["_runtime_context_cache"]) == 2
