"""Anima Artist Mixer Plus for Diffusers-like Anima pipelines.

This file implements the same high-level syntax as the ComfyUI node:

    {@artist1:0.8, @artist2:1.2}
    @artist:style:0.7
    @artist:pose:0.4
    @artist:[style:1.0, face:0.4, eyes pose:0.7]

It patches Anima/MiniTrainDIT cross-attention modules and routes every artist to
the active Anima transformer depth according to the requested component. The
original 28-block layout and the expanded 40-block Anima 2.9B layout are mapped
semantically so inserted blocks inherit the role of their source block. The script
is designed as a safe adapter: if your Diffusers Anima implementation uses different
module names, pass custom `blocks_getter` / `cross_attn_getter` callbacks.

Important: non-cross-attention elements are not directly re-encoded in this
Diffusers version. Artist text affects the model through per-layer cross-attn
routing, because generic Diffusers modules do not expose a stable ComfyUI-style
object patch interface for self_attn/mlp/adaln across all forks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FUSION_INTERPOLATE = "interpolate"
FUSION_BASE_PRESERVE = "base_preserve"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_LOWRANK_AVG = "lowrank_avg"
MAX_ARTISTS = 32

_COMPONENT_ALIASES = {
    "all": "global", "overall": "global", "mix": "global", "global": "global",
    "style": "style", "artstyle": "style", "default_style": "style", "画風": "style",
    "paint": "paint", "painting": "paint", "render": "paint", "rendering": "paint", "塗り": "paint",
    "color": "color", "colour": "color", "saturation": "color", "contrast": "color", "色": "color",
    "texture": "texture", "line": "texture", "lines": "texture", "outline": "texture", "線": "texture",
    "face": "face", "顔": "face",
    "eyes": "eyes", "eye": "eyes", "目": "eyes",
    "mouth": "mouth", "nose": "mouth", "jaw": "mouth", "lower_face": "mouth", "口": "mouth", "鼻": "mouth",
    "expression": "expression", "expr": "expression", "表情": "expression",
    "age": "age", "年齢": "age",
    "body": "body", "physique": "body", "体型": "body", "身体": "body",
    "pose": "pose", "posing": "pose", "ポーズ": "pose",
    "hands": "hands", "hand": "hands", "fingers": "hands", "finger": "hands", "手": "hands", "指": "hands",
    "clothes": "clothes", "clothing": "clothes", "costume": "clothes", "衣装": "clothes", "服": "clothes",
    "background": "background", "bg": "background", "背景": "background",
    "light": "light", "lighting": "lighting", "gloss": "light", "光": "light",
}


def _layer_range(lo: int, hi: int, value: float = 1.0) -> Dict[int, float]:
    return {i: float(value) for i in range(int(lo), int(hi) + 1)}


def _merge_layer_maps(*maps: Dict[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for mp in maps:
        for k, v in mp.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


_BASE_COMPONENT_LAYER_MAP: Dict[str, Dict[int, float]] = {
    "global": _merge_layer_maps(_layer_range(0, 27, 1.0)),
    "style": _merge_layer_maps(_layer_range(0, 27, 0.20), _layer_range(14, 21, 0.45), _layer_range(22, 24, 0.95), _layer_range(25, 27, 1.15)),
    "paint": _merge_layer_maps(_layer_range(14, 17, 0.35), _layer_range(18, 21, 0.70), _layer_range(22, 27, 1.10)),
    "color": _merge_layer_maps(_layer_range(20, 24, 0.80), _layer_range(25, 27, 1.20)),
    "texture": _merge_layer_maps(_layer_range(0, 8, 0.25), _layer_range(22, 27, 1.00)),
    "light": _merge_layer_maps(_layer_range(22, 24, 0.85), _layer_range(25, 27, 0.65)),
    "face": _merge_layer_maps(_layer_range(10, 13, 1.00), _layer_range(14, 17, 0.80), _layer_range(18, 21, 0.75)),
    "eyes": _merge_layer_maps(_layer_range(12, 13, 0.35), _layer_range(14, 17, 1.20), _layer_range(18, 19, 0.25)),
    "mouth": _merge_layer_maps(_layer_range(18, 21, 1.10), _layer_range(14, 17, 0.25)),
    "expression": _merge_layer_maps(_layer_range(14, 21, 1.00), _layer_range(10, 13, 0.35)),
    "age": _merge_layer_maps(_layer_range(10, 13, 0.70), _layer_range(14, 21, 1.00)),
    "body": _merge_layer_maps(_layer_range(4, 8, 1.00), _layer_range(9, 13, 0.85), _layer_range(16, 21, 0.35)),
    "pose": _merge_layer_maps(_layer_range(6, 8, 0.85), _layer_range(9, 13, 1.10), _layer_range(14, 15, 0.55)),
    "hands": _merge_layer_maps(_layer_range(0, 8, 1.10), _layer_range(9, 18, 0.35)),
    "clothes": _merge_layer_maps(_layer_range(3, 8, 1.00), _layer_range(9, 13, 0.55), _layer_range(22, 24, 0.30)),
    "background": _merge_layer_maps(_layer_range(2, 3, 0.80), _layer_range(22, 24, 1.00), _layer_range(25, 27, 0.45)),
}


# Anima 2.9B expands the original 28 main blocks to 40. The inserted blocks are
# derived from the source blocks listed in expanded_manifest, so component
# routing inherits the source block's semantic weight instead of stopping at L27.
_ANIMA_29B_INSERTED_TO_SOURCE: Dict[int, int] = {
    2: 1, 5: 3, 8: 5, 11: 7, 14: 9, 17: 11,
    21: 14, 24: 16, 27: 18, 30: 20, 33: 22, 36: 24,
}


def _expanded_29b_source_map() -> Dict[int, int]:
    source_for_block: Dict[int, int] = {}
    source_index = 0
    for block_index in range(40):
        if block_index in _ANIMA_29B_INSERTED_TO_SOURCE:
            source_for_block[block_index] = _ANIMA_29B_INSERTED_TO_SOURCE[block_index]
        else:
            source_for_block[block_index] = source_index
            source_index += 1
    if source_index != 28:
        raise RuntimeError("Invalid Anima 2.9B block mapping")
    return source_for_block


_ANIMA_29B_SOURCE_FOR_BLOCK = _expanded_29b_source_map()


def _component_layer_map_for_depth(num_blocks: int) -> Dict[str, Dict[int, float]]:
    """Project the original 28-block component map onto the active architecture."""
    if num_blocks <= 0:
        return {name: {} for name in _BASE_COMPONENT_LAYER_MAP}
    if num_blocks == 28:
        return _BASE_COMPONENT_LAYER_MAP

    if num_blocks == 40:
        source_for_block = _ANIMA_29B_SOURCE_FOR_BLOCK
    else:
        # Future-compatible fallback: preserve relative depth roles rather than
        # silently leaving all blocks above L27 inactive.
        if num_blocks == 1:
            source_for_block = {0: 0}
        else:
            source_for_block = {
                i: int(round(i * 27.0 / float(num_blocks - 1)))
                for i in range(num_blocks)
            }

    projected: Dict[str, Dict[int, float]] = {}
    for component, baseline in _BASE_COMPONENT_LAYER_MAP.items():
        projected[component] = {
            block_index: float(baseline.get(source_index, 0.0))
            for block_index, source_index in source_for_block.items()
            if float(baseline.get(source_index, 0.0)) != 0.0
        }
    return projected


@dataclass
class ArtistSpec:
    name: str
    components: Dict[str, float]
    explicit: bool = False


def _split_top_level(text: str, delimiter: str = ',') -> List[str]:
    out, buf = [], []
    depth = 0
    quote: Optional[str] = None
    pairs = {'[': ']', '{': '}', '(': ')'}
    closing = set(pairs.values())
    for ch in str(text or ''):
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
            continue
        if ch in pairs:
            depth += 1
            buf.append(ch)
            continue
        if ch in closing:
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == delimiter and depth == 0:
            item = ''.join(buf).strip()
            if item:
                out.append(item)
            buf = []
        else:
            buf.append(ch)
    item = ''.join(buf).strip()
    if item:
        out.append(item)
    return out


def _find_top_level_char(text: str, target: str) -> int:
    depth = 0
    quote: Optional[str] = None
    pairs = {'[': ']', '{': '}', '(': ')'}
    closing = set(pairs.values())
    for idx, ch in enumerate(str(text or '')):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            continue
        if ch in pairs:
            depth += 1
            continue
        if ch in closing:
            depth = max(0, depth - 1)
            continue
        if ch == target and depth == 0:
            return idx
    return -1


def _strip_outer_group(text: str) -> str:
    s = str(text or '').strip()
    if not (s.startswith('{') and s.endswith('}')):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
    return s[1:-1].strip()


def _safe_float(value: Any, default: float = 1.0, lo: float = 0.0, hi: float = 4.0) -> float:
    try:
        v = float(str(value).strip())
    except Exception:
        return float(default)
    return max(float(lo), min(float(hi), v))


def _normalize_component_name(name: str) -> Optional[str]:
    key = str(name or '').strip().lower().replace('-', '_')
    return _COMPONENT_ALIASES.get(key, key if key in _BASE_COMPONENT_LAYER_MAP else None)


def _split_component_names(text: str) -> List[str]:
    raw = str(text or '').replace('+', ' ').replace('|', ' ').replace('/', ' ')
    names: List[str] = []
    for part in raw.split():
        comp = _normalize_component_name(part)
        if comp and comp not in names:
            names.append(comp)
    return names


def _parse_component_items(inner: str) -> Dict[str, float]:
    comps: Dict[str, float] = {}
    for item in _split_top_level(inner, ','):
        pos = _find_top_level_char(item, ':')
        if pos < 0:
            for comp in _split_component_names(item):
                comps[comp] = comps.get(comp, 0.0) + 1.0
            continue
        lhs, rhs = item[:pos].strip(), item[pos + 1:].strip()
        ratio = _safe_float(rhs, 1.0)
        for comp in _split_component_names(lhs):
            comps[comp] = comps.get(comp, 0.0) + ratio
    return comps


def _parse_old_weight_syntax(text: str) -> Tuple[str, float, bool]:
    s = str(text or '').strip()
    if '::' not in s:
        return s, 1.0, False
    head = s[2:] if s.startswith('::') else s
    if '::' not in head:
        return s, 1.0, False
    name_part, _, w_part = head.rpartition('::')
    try:
        return name_part.strip(), _safe_float(w_part, 1.0), True
    except Exception:
        return s, 1.0, False


def parse_artist_mixer_syntax(chain: str) -> List[ArtistSpec]:
    body = _strip_outer_group(chain)
    specs: List[ArtistSpec] = []
    for entry in _split_top_level(body, ','):
        s, outer_weight, explicit_old = _parse_old_weight_syntax(entry)
        if not s:
            continue
        colon = _find_top_level_char(s, ':')
        if colon >= 0:
            name, rest = s[:colon].strip(), s[colon + 1:].strip()
            if rest.startswith('[') and rest.endswith(']'):
                comps = _parse_component_items(rest[1:-1])
                comps = {k: v * outer_weight for k, v in comps.items() if k in _BASE_COMPONENT_LAYER_MAP}
                specs.append(ArtistSpec(name=name, components=comps or {"global": outer_weight}, explicit=True))
                continue
            parts = _split_top_level(s, ':')
            if len(parts) == 2:
                comp = _normalize_component_name(parts[1].strip())
                if comp:
                    specs.append(ArtistSpec(name=parts[0].strip(), components={comp: outer_weight}, explicit=True))
                else:
                    specs.append(ArtistSpec(name=parts[0].strip(), components={"global": _safe_float(parts[1], 1.0) * outer_weight}, explicit=True))
                continue
            if len(parts) >= 3:
                ratio = _safe_float(parts[-1], 1.0) * outer_weight
                comp_text = ':'.join(parts[1:-1]).strip()
                comps = {comp: ratio for comp in _split_component_names(comp_text)}
                specs.append(ArtistSpec(name=parts[0].strip(), components=comps or {"global": ratio}, explicit=True))
                continue
        specs.append(ArtistSpec(name=s, components={"global": outer_weight}, explicit=explicit_old))
    return specs[:MAX_ARTISTS]


def build_layer_artist_weights(specs: Sequence[ArtistSpec], num_blocks: int) -> Dict[int, List[float]]:
    layer_artist_weights = {i: [0.0] * len(specs) for i in range(num_blocks)}
    component_layer_map = _component_layer_map_for_depth(num_blocks)
    for artist_idx, spec in enumerate(specs):
        for comp, ratio in spec.components.items():
            for layer, lw in component_layer_map.get(comp, {}).items():
                if 0 <= layer < num_blocks:
                    layer_artist_weights[layer][artist_idx] += float(ratio) * float(lw)
    for layer in range(num_blocks):
        layer_artist_weights[layer] = [max(0.0, min(4.0, float(w))) for w in layer_artist_weights[layer]]
    return layer_artist_weights


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = sum(abs(float(w)) for w in weights)
    if total <= 1e-8:
        return [1.0 / max(1, len(weights))] * len(weights)
    return [float(w) / total for w in weights]


def _broadcast_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, *t.shape[1:])
    if batch_size % t.shape[0] == 0:
        return t.repeat(batch_size // t.shape[0], *([1] * (t.dim() - 1)))
    return t[:1].expand(batch_size, *t.shape[1:])


def _project_perpendicular(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    base_norm_sq = (base * base).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (delta * base).sum(dim=-1, keepdim=True) / base_norm_sq
    return delta - proj_coef * base


def _replace_first_tensor(container: Any, new_tensor: torch.Tensor) -> Any:
    if torch.is_tensor(container):
        return new_tensor
    if isinstance(container, tuple) and container and torch.is_tensor(container[0]):
        return (new_tensor,) + tuple(container[1:])
    if isinstance(container, list) and container and torch.is_tensor(container[0]):
        out = list(container)
        out[0] = new_tensor
        return out
    return new_tensor


class DiffusersCrossAttnMixerWrapper(nn.Module):
    def __init__(self, original: nn.Module, state: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.original = original
        self.state = state
        self.layer_idx = int(layer_idx)

    def _get_context(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str, int]:
        for key in ("encoder_hidden_states", "context", "encoder_states"):
            v = kwargs.get(key)
            if torch.is_tensor(v):
                return v, key, -1
        if len(args) >= 2 and torch.is_tensor(args[1]):
            return args[1], "", 1
        return None, "", -1

    def _call_with_context(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], key: str, pos: int, ctx: torch.Tensor) -> Any:
        if key:
            kw = dict(kwargs)
            kw[key] = ctx
            return self.original(*args, **kw)
        if pos >= 0:
            ar = list(args)
            ar[pos] = ctx
            return self.original(*ar, **kwargs)
        return self.original(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.state.get("enabled", True):
            return self.original(*args, **kwargs)
        context, key, pos = self._get_context(args, kwargs)
        if context is None:
            return self.original(*args, **kwargs)
        weights = self.state["layer_artist_weights"].get(self.layer_idx, [])
        if not weights or max(abs(float(w)) for w in weights) <= 1e-8:
            return self.original(*args, **kwargs)

        base_raw = self.original(*args, **kwargs)
        base = base_raw[0] if isinstance(base_raw, (tuple, list)) and torch.is_tensor(base_raw[0]) else base_raw
        if not torch.is_tensor(base):
            return base_raw

        artist_contexts: List[torch.Tensor] = self.state["artist_contexts"]
        if self.state.get("normalize_weights", True):
            ws = _normalize_weights(weights)
        else:
            ws = list(weights)

        outs: List[torch.Tensor] = []
        bsz = context.shape[0]
        for ctx in artist_contexts:
            ctx_b = _broadcast_batch(ctx.to(device=context.device, dtype=context.dtype), bsz)
            out_raw = self._call_with_context(args, kwargs, key, pos, ctx_b)
            out = out_raw[0] if isinstance(out_raw, (tuple, list)) and torch.is_tensor(out_raw[0]) else out_raw
            if torch.is_tensor(out):
                outs.append(out)
        if not outs:
            return base_raw

        combine_mode = self.state.get("combine_mode", COMBINE_OUTPUT_AVG)
        if combine_mode == COMBINE_LOWRANK_AVG and len(outs) >= 2:
            n = len(outs)
            k = max(1, min(int(self.state.get("lowrank_k", 1)), n))
            A = torch.stack(outs, dim=0).to(torch.float32)
            base32 = base.to(torch.float32).unsqueeze(0)
            delta = A - base32
            D = delta.reshape(n, -1)
            if k < n:
                try:
                    U, S, V = torch.svd_lowrank(D, q=k, niter=2)
                    D = U @ torch.diag(S) @ V.transpose(-1, -2)
                except Exception as e:
                    logger.warning("[DiffusersAnimaArtistMixer] SVD failed; fallback to output_avg: %s", e)
            w_t = torch.tensor(ws, device=D.device, dtype=D.dtype).view(n, 1)
            delta_avg = (D * w_t).sum(dim=0).reshape_as(base).to(base.dtype)
            artist_total = base + delta_avg
        else:
            artist_total = None
            for out, w in zip(outs, ws):
                artist_total = out * float(w) if artist_total is None else artist_total + out * float(w)

        strength = float(self.state.get("strength", 1.0))
        fusion_mode = self.state.get("fusion_mode", FUSION_INTERPOLATE)
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = _project_perpendicular(artist_total - base, base)
            mixed = base + strength * delta
        else:
            mixed = base * (1.0 - strength) + artist_total * strength
        return _replace_first_tensor(base_raw, mixed)


def _default_find_transformer(pipe: Any) -> Any:
    for name in ("transformer", "model", "unet", "dit", "diffusion_model"):
        obj = getattr(pipe, name, None)
        if obj is not None:
            return obj
    raise AttributeError("Could not find transformer/model/unet/dit on pipeline. Pass a custom blocks_getter.")


def _default_get_blocks(pipe: Any) -> Sequence[Any]:
    root = _default_find_transformer(pipe)
    candidates = [root]
    core = getattr(root, "core", None)
    if core is not None:
        candidates.append(core)
    for candidate in candidates:
        for name in ("blocks", "transformer_blocks", "layers", "joint_transformer_blocks"):
            blocks = getattr(candidate, name, None)
            if blocks is not None and len(blocks) > 0:
                return blocks
    raise AttributeError(
        "Could not find blocks/transformer_blocks/layers on pipeline transformer "
        "or transformer.core. Pass blocks_getter."
    )


def _default_cross_attn_getter(block: Any) -> Tuple[str, nn.Module]:
    for name in ("cross_attn", "attn2", "cross_attention"):
        obj = getattr(block, name, None)
        if isinstance(obj, nn.Module):
            return name, obj
    raise AttributeError("Could not find cross_attn/attn2/cross_attention on block. Pass cross_attn_getter.")


def default_encode_artist(pipe: Any, text: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Best-effort Diffusers prompt encoder.

    Some Anima forks return multiple values from encode_prompt. This function picks
    the first 3D tensor, which is normally prompt/encoder hidden states. If your
    pipeline needs t5 ids/weights preprocessing, pass your own encode_artist_fn.
    """
    if device is None:
        try:
            device = next(pipe.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipe, "encode_prompt"):
        try:
            out = pipe.encode_prompt(
                prompt=text,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        except TypeError:
            out = pipe.encode_prompt(text)
        candidates = out if isinstance(out, (tuple, list)) else (out,)
        for item in candidates:
            if torch.is_tensor(item) and item.dim() >= 3:
                return item.to(device=device, dtype=dtype or item.dtype)
    if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
        tok = pipe.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            out = pipe.text_encoder(**tok)
        hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        return hidden.to(device=device, dtype=dtype or hidden.dtype)
    raise AttributeError("No usable encode_prompt/tokenizer+text_encoder found. Pass encode_artist_fn.")


class DiffusersAnimaArtistMixer:
    def __init__(
        self,
        pipe: Any,
        blocks_getter: Callable[[Any], Sequence[Any]] = _default_get_blocks,
        cross_attn_getter: Callable[[Any], Tuple[str, nn.Module]] = _default_cross_attn_getter,
    ) -> None:
        self.pipe = pipe
        self.blocks_getter = blocks_getter
        self.cross_attn_getter = cross_attn_getter
        self._patched: List[Tuple[Any, str, nn.Module]] = []
        self.state: Dict[str, Any] = {}

    def encode_specs(
        self,
        syntax: str,
        base_prompt: str = "",
        encode_artist_fn: Callable[[Any, str], torch.Tensor] = default_encode_artist,
    ) -> Tuple[List[ArtistSpec], List[torch.Tensor]]:
        specs = parse_artist_mixer_syntax(syntax)
        contexts: List[torch.Tensor] = []
        for spec in specs:
            text = f"{spec.name}\n{base_prompt}" if base_prompt else spec.name
            contexts.append(encode_artist_fn(self.pipe, text))
        return specs, contexts

    def install(
        self,
        syntax: str,
        base_prompt: str = "",
        encode_artist_fn: Callable[[Any, str], torch.Tensor] = default_encode_artist,
        strength: float = 1.0,
        normalize_weights: bool = True,
        fusion_mode: str = FUSION_INTERPOLATE,
        combine_mode: str = COMBINE_OUTPUT_AVG,
        lowrank_k: int = 1,
        start_block: int = 0,
        end_block: int = -1,
        layer_filter: Optional[Iterable[int]] = None,
    ) -> "DiffusersAnimaArtistMixer":
        self.uninstall()
        specs, contexts = self.encode_specs(syntax, base_prompt, encode_artist_fn)
        if not specs:
            logger.warning("[DiffusersAnimaArtistMixer] no artist specs parsed; nothing patched")
            return self
        blocks = self.blocks_getter(self.pipe)
        num_blocks = len(blocks)
        weights = build_layer_artist_weights(specs, num_blocks)
        if any(spec.explicit for spec in specs):
            normalize_weights = False

        if layer_filter is not None:
            target_blocks = [i for i in layer_filter if 0 <= int(i) < num_blocks]
        else:
            sb = max(0, int(start_block))
            eb = num_blocks - 1 if int(end_block) < 0 else min(num_blocks - 1, int(end_block))
            target_blocks = list(range(sb, eb + 1))
        active = {i for i, ws in weights.items() if max(abs(float(w)) for w in ws) > 1e-8}
        target_blocks = [i for i in target_blocks if i in active]

        self.state = {
            "enabled": True,
            "artist_contexts": contexts,
            "component_specs": specs,
            "layer_artist_weights": weights,
            "strength": float(strength),
            "normalize_weights": bool(normalize_weights),
            "fusion_mode": fusion_mode,
            "combine_mode": combine_mode,
            "lowrank_k": int(lowrank_k),
        }

        for i in target_blocks:
            block = blocks[i]
            try:
                attr_name, original = self.cross_attn_getter(block)
            except Exception as e:
                logger.debug("[DiffusersAnimaArtistMixer] skip block %d: %s", i, e)
                continue
            wrapper = DiffusersCrossAttnMixerWrapper(original, self.state, i)
            setattr(block, attr_name, wrapper)
            self._patched.append((block, attr_name, original))
        logger.info("[DiffusersAnimaArtistMixer] patched %d blocks", len(self._patched))
        return self

    def uninstall(self) -> None:
        while self._patched:
            block, attr_name, original = self._patched.pop()
            setattr(block, attr_name, original)
        self.state = {}

    def __enter__(self) -> "DiffusersAnimaArtistMixer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.uninstall()


__all__ = [
    "ArtistSpec",
    "DiffusersAnimaArtistMixer",
    "parse_artist_mixer_syntax",
    "default_encode_artist",
    "FUSION_INTERPOLATE",
    "FUSION_BASE_PRESERVE",
    "COMBINE_OUTPUT_AVG",
    "COMBINE_LOWRANK_AVG",
]
