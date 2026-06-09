# Anima Artist Mixer + sd_embed Anima encoder integration

## Files

- `embedding_funcs_artist_mixer_integrated.py`  
  Drop-in replacement for your current `embedding_funcs.py`. It keeps the existing Anima weighted encoder and adds optional Artist Mixer support.

- `anima_artist_mixer_plus_diffusers.py`  
  Runtime transformer patcher used by the integrated encoder. Put it next to `embedding_funcs_artist_mixer_integrated.py` or on `PYTHONPATH`.

## Basic usage

```python
from embedding_funcs_artist_mixer_integrated import (
    get_weighted_text_embeddings_anima,
    uninstall_anima_artist_mixer,
)

prompt = """
{@vlizz:[style:1.0, pose:0.4], @ashraely:[style:0.6, body:0.8]},
1girl, dynamic pose, cinematic lighting
"""

pos, neg = get_weighted_text_embeddings_anima(
    pipe,
    prompt=prompt,
    neg_prompt="low quality, blurry",
    enable_artist_mixer=True,
    artist_mixer_strength=1.0,
    artist_mixer_fusion_mode="interpolate",
    artist_mixer_combine_mode="output_avg",
)

image = pipe(
    prompt_embeds=pos,
    negative_prompt_embeds=neg,
    # other generation args...
).images[0]

uninstall_anima_artist_mixer(pipe)
```

## Keeping mixer syntax outside the visible prompt

```python
pos, neg = get_weighted_text_embeddings_anima(
    pipe,
    prompt="1girl, dynamic pose, cinematic lighting",
    neg_prompt="low quality, blurry",
    artist_mixer="{@vlizz:[style:1.0, pose:0.4], @ashraely:[style:0.6, body:0.8]}",
    artist_mixer_strength=1.0,
)

image = pipe(prompt_embeds=pos, negative_prompt_embeds=neg).images[0]
uninstall_anima_artist_mixer(pipe)
```

## Notes

- `enable_artist_mixer=True` extracts top-level mixer syntax from the positive prompt before normal Anima text encoding.
- Plain normal Anima artist tags such as `@artist` are not removed unless they use mixer syntax like `@artist:[style:0.7]` or `@artist:eyes:1.2`.
- The mixer is a transformer patch, so it must remain installed until the actual `pipe(...)` generation call finishes.
- Call `uninstall_anima_artist_mixer(pipe)` after generation to restore the pipeline.
- If your Diffusers Anima fork uses different block/cross-attention attribute names, edit `anima_artist_mixer_plus_diffusers.py` and customize `_default_get_blocks` / `_default_cross_attn_getter`.
