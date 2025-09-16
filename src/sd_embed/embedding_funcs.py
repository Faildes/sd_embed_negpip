## -----------------------------------------------------------------------------
# Generate unlimited size prompt with weighting for SD3&SDXL&SD15
# If you use sd_embed in your research, please cite the following work:
# 
# ```
# @misc{sd_embed_2024,
#   author       = {Shudong Zhu(Andrew Zhu)},
#   title        = {Long Prompt Weighted Stable Diffusion Embedding},
#   howpublished = {\url{https://github.com/xhinker/sd_embed}},
#   year         = {2024},
# }
# ```
# Author: Andrew Zhu
# Book: Using Stable Diffusion with Python, https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373
# Github: https://github.com/xhinker
# Medium: https://medium.com/@xhinker
## -----------------------------------------------------------------------------

from sd_embed.prompt_parser import parse_prompt_attention
from transformers import CLIPTokenizer,T5EncoderModel,T5Tokenizer
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
from diffusers import StableDiffusion3Pipeline
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
import math
import traceback
from diffusers import FluxPipeline
from typing import Any, List, Optional, Tuple
import gc
import logging
import typing

logger = logging.getLogger(__name__)

def _split_pos_neg_weights(weights: list):
    neg_mask = [w < 0 for w in weights]
    pos_weights = [ (1.0 if w < 0 else w) for w in weights ]
    abs_neg_weights = [ (abs(w) if w < 0 else 0.0) for w in weights ]
    return neg_mask, pos_weights, abs_neg_weights

def _apply_token_weights_inplace(token_embedding: torch.Tensor, weights_tensor: torch.Tensor):
    for j in range(len(weights_tensor)):
        w = float(weights_tensor[j])
        if w != 1.0 and w != 0.0:
            token_embedding[j] = token_embedding[j] * w
    return token_embedding

def _apply_method2_linear_inplace(emb: torch.Tensor, weights_1d: torch.Tensor) -> None:
    """
    Method-2 interpolation only (no tanh):
        emb[j] = base + (emb[j] - base) * w
    """
    base = emb[-1]
    for j in range(len(weights_1d)):
        w = float(weights_1d[j])
        if w == 1.0:
            continue
        emb[j] = base + (emb[j] - base) * w

def _apply_method2_tanh_inplace(emb: torch.Tensor, weights_1d: torch.Tensor) -> None:
    """
    Original refiner-style: tanh-like mapping first, then method-2 interpolation:
        ow = w - 1
        tanh_weight = (exp(ow)/(exp(ow)+1) - 0.5) * 2
        w' = 1 + tanh_weight
        emb[j] = base + (emb[j] - base) * w'
    """
    base = emb[-1]
    for j in range(len(weights_1d)):
        w = float(weights_1d[j])
        if w == 1.0:
            continue
        ow = w - 1.0
        tanh_weight = (math.exp(ow) / (math.exp(ow) + 1.0) - 0.5) * 2.0
        w_mapped = 1.0 + tanh_weight
        emb[j] = base + (emb[j] - base) * w_mapped

def _negpip_dual_apply(prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor, alpha: float = 1.0):
    new_prompt = prompt_embeds - alpha * negative_prompt_embeds
    new_negative = negative_prompt_embeds + negative_prompt_embeds
    return new_prompt, new_negative

def _negpip_dual_apply_pooled(pp: Optional[torch.Tensor], np: Optional[torch.Tensor], alpha: float = 1.0):
    if pp is None or np is None:
        return pp, np
    return pp - alpha * np, np + np

def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer
    , prompt: str = None
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    
    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights
            
    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list) 
            A list contains the correspodent weight of token ids
    
    Example:
        import torch
        from diffusers_plus.tools.sd_embeddings import get_prompts_tokens_with_weights
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(
            word
            , truncation = False        # so that tokenize whatever length prompt
        ).input_ids[1:-1]
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def get_prompts_tokens_with_weights_t5(
    t5_tokenizer: T5Tokenizer
    , prompt: str
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = t5_tokenizer(
            word
            , truncation            = False        # so that tokenize whatever length prompt
            , add_special_tokens    = True
        ).input_ids
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def group_tokens_and_weights(
    token_ids: list
    , weights: list
    , pad_last_block = True
):
    """
    Produce tokens and weights in groups and pad the missing tokens
    
    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)
    
    Example:
        from diffusers_plus.tools.sd_embeddings import group_tokens_and_weights
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos,eos = 49406,49407
    
    # this will be a 2d list 
    new_token_ids = []
    new_weights   = []  
    while len(token_ids) >= 75:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]
        
        # extract token ids and weights
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights   = [1.0] + head_75_weights + [1.0]
        
        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)
    
    # padding the left
    if len(token_ids) > 0:
        padding_len         = 75 - len(token_ids) if pad_last_block else 0
        
        temp_77_token_ids   = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)
        
        temp_77_weights     = [1.0] + weights   + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)
        
    return new_token_ids, new_weights

def get_prompt_hidden_states(
    prompt_embeds: torch.Tensor
    , final_layer_index: int
    , clip_skip: Optional[int]   = None
):
    if clip_skip is None:
        return prompt_embeds.hidden_states[-(final_layer_index)]
    return prompt_embeds.hidden_states[-(clip_skip + final_layer_index)]

def get_prompt_hidden_states_sdxl(
    prompt_embeds: torch.Tensor
    , clip_skip: Optional[int]   = None
):
    # "'2' because SDXL always indexes from the penultimate layer."
    # (https://github.com/huggingface/diffusers/blob/9f5ad1db4197d6c2b503dd5fa3ef4dbec12a4f96/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L436)
    return get_prompt_hidden_states(prompt_embeds, 2, clip_skip=clip_skip)

def get_prompt_hidden_states_s_cascade(
    prompt_embeds: torch.Tensor
    , clip_skip: Optional[int]   = None
):
    # 
    return get_prompt_hidden_states(prompt_embeds, 1, clip_skip=clip_skip)

def get_prompt_hidden_states_sd3(
    prompt_embeds: torch.Tensor
    , clip_skip: Optional[int]   = None
):
    # SD3 seems to use the same layer index as SDXL
    return get_prompt_hidden_states(prompt_embeds, 2, clip_skip=clip_skip)

def get_weighted_text_embeddings_sd15(
    pipe: StableDiffusionPipeline
    , prompt : str      = ""
    , neg_prompt: str   = ""
    , pad_last_block    = False
    , clip_skip:int     = 0
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for Stable Diffusion v1.5, and extends the behavior with NegPiP routing + dual composition.

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * In the positive prompt: tokens with negative weights are removed from the
            positive stream and their absolute weights are sent to the "neg-from-pos" stream.
        * In the negative prompt: tokens with negative weights are removed from the
            negative stream and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition:
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_embeds            = prompt_pre - neg_total
        * final negative_prompt_embeds   = neg_total + neg_total
        (Per-token weighting is applied linearly to the hidden states.)

    Tokenization & grouping:
    - Prompts are parsed with weights, tokenized, and split into groups of 77 tokens
        ([BOS] + 75 tokens + [EOS]). If needed, the shorter side is padded to keep
        positive/negative parity. The last block can be optionally padded with EOS.

    Args:
        pipe (StableDiffusionPipeline)
            A diffusers StableDiffusionPipeline (SD 1.5). Must provide a CLIP tokenizer
            and text encoder compatible with SD 1.5.
        prompt (str)
            A prompt string with weights. Example: "a (white:1.2) cat, (fur:-0.5)".
            Negative weights here will be routed to the negative stream as "neg-from-pos".
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here will be routed to the positive stream as "pos-from-neg".
        pad_last_block (bool)
            If True, pads the tail block to 75 tokens with EOS so each group forms 77
            with BOS/EOS. If False, the last group is left unpadded.
        clip_skip (int)
            Number of final CLIP layers to skip (0 = use default). When > 0, the last
            `clip_skip` encoder layers are omitted before extracting hidden states.

    Returns:
        prompt_embeds (torch.Tensor)
            The final positive embeddings after NegPiP dual composition. Shape: [1, T*, C].
        neg_prompt_embeds (torch.Tensor)
            The final negative embeddings after NegPiP dual composition. Shape: [1, T*, C].

    Example:
        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda:0")

        prompt = "a (white:1.2) cat, soft light, (fur:-0.4)"
        neg_prompt = "(blur:1.2), (artifact:-0.6)"  # '-0.6' will be routed to positive

        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_sd15(
            pipe=pipe,
            prompt=prompt,
            neg_prompt=neg_prompt,
            pad_last_block=False,
            clip_skip=0
        )

        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            generator=torch.Generator(pipe.device).manual_seed(2)
        ).images[0]
    """
    original_clip_layers = pipe.text_encoder.text_model.encoder.layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]
    
    eos = pipe.tokenizer.eos_token_id
    
    # --- 1) tokenize and get weights ---
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

    # --- 2) pad to equal length ---
    pt_len, nt_len = len(prompt_tokens), len(neg_prompt_tokens)
    pad = abs(pt_len - nt_len)
    if pt_len > nt_len:
        neg_prompt_tokens  += [eos] * pad
        neg_prompt_weights += [1.0] * pad
    elif nt_len > pt_len:
        prompt_tokens  += [eos] * pad
        prompt_weights += [1.0] * pad

    # --- 3) split weights: positive/negative transfer ---
    _, pos_w_p, abs_neg_w_p = _split_pos_neg_weights(prompt_weights)   # pos side
    _, pos_w_n, pos_from_n  = _split_pos_neg_weights(neg_prompt_weights)  # neg side

    # --- 4) group into 77-token blocks ---
    p_groups, p_posw_groups = group_tokens_and_weights(prompt_tokens.copy(), pos_w_p.copy(), pad_last_block)
    _, p_negfrompos_groups  = group_tokens_and_weights(prompt_tokens.copy(), abs_neg_w_p.copy(), pad_last_block)
    n_groups, n_posw_groups = group_tokens_and_weights(neg_prompt_tokens.copy(), pos_w_n.copy(), pad_last_block)
    _, n_posfromneg_groups  = group_tokens_and_weights(neg_prompt_tokens.copy(), pos_from_n.copy(), pad_last_block)

    # --- 5) embed each group ---
    pos_base, neg_base, neg_from_pos, pos_from_neg = [], [], [], []
    for i in range(len(p_groups)):
        # positive base
        tok = torch.tensor([p_groups[i]], dtype=torch.long, device=pipe.device)
        w   = torch.tensor(p_posw_groups[i], dtype=torch.float16, device=pipe.device)
        emb = pipe.text_encoder(tok)[0].squeeze(0)
        _apply_token_weights_inplace(emb, w)
        pos_base.append(emb.unsqueeze(0))

        # positive side: extract negative words (abs weight)
        w_absneg = torch.tensor(p_negfrompos_groups[i], dtype=torch.float16, device=pipe.device)
        emb_abs  = pipe.text_encoder(tok)[0].squeeze(0)
        _apply_token_weights_inplace(emb_abs, w_absneg)
        neg_from_pos.append(emb_abs.unsqueeze(0))

        # negative base
        tok_n = torch.tensor([n_groups[i]], dtype=torch.long, device=pipe.device)
        w_n   = torch.tensor(n_posw_groups[i], dtype=torch.float16, device=pipe.device)
        emb_n = pipe.text_encoder(tok_n)[0].squeeze(0)
        _apply_token_weights_inplace(emb_n, w_n)
        neg_base.append(emb_n.unsqueeze(0))

        # negative side: words with negative weights moved to positive
        w_posfromn = torch.tensor(n_posfromneg_groups[i], dtype=torch.float16, device=pipe.device)
        emb_posn   = pipe.text_encoder(tok_n)[0].squeeze(0)
        _apply_token_weights_inplace(emb_posn, w_posfromn)
        pos_from_neg.append(emb_posn.unsqueeze(0))

    # --- 6) concatenate embeddings ---
    prompt_base       = torch.cat(pos_base,     dim=1)
    neg_base_embeds   = torch.cat(neg_base,     dim=1)
    neg_from_pos_embs = torch.cat(neg_from_pos, dim=1)
    pos_from_neg_embs = torch.cat(pos_from_neg, dim=1)

    # --- 7) merge routed parts ---
    prompt_pre = prompt_base + pos_from_neg_embs
    neg_total  = neg_base_embeds + neg_from_pos_embs

    # --- 8) NegPiP dual apply (alpha=1.0) ---
    prompt_embeds, neg_prompt_embeds = _negpip_dual_apply(prompt_pre, neg_total, alpha=1.0)

    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers

    return prompt_embeds, neg_prompt_embeds

# Dynamically adjust the LoRA scale, determining which text_encoder attributes the pipeline has (if any) dynamically as well
# BEGIN: based on https://github.com/huggingface/diffusers/blob/9f5ad1db4197d6c2b503dd5fa3ef4dbec12a4f96/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L367
def dynamically_scale_lora_layers(
    pipe: DiffusionPipeline
    , lora_scale: Optional[float]  = None
):
    if lora_scale is None:
        return

    text_encoder_attributes: List[str] = [ "text_encoder", "text_encoder_2", "text_encoder_3" ]
    for attribute_name in text_encoder_attributes:
        if hasattr(pipe, attribute_name):
            try:
                encoder_attribute: Optional[Any] = getattr(pipe, attribute_name)
                if encoder_attribute is not None:
                    if not USE_PEFT_BACKEND:
                        adjust_lora_scale_text_encoder(encoder_attribute, lora_scale)
                    else:
                        scale_lora_layers(encoder_attribute, lora_scale)
            except Exception as e:
                logger.error(f"Couldn't apply LoRA scale to pipeline attribute {attribute_name}: {e}\n{traceback.format_exc()}")
                    
# END: based on https://github.com/huggingface/diffusers/blob/9f5ad1db4197d6c2b503dd5fa3ef4dbec12a4f96/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L367

# Undo the changes caused by scale_lora_layers, determining which text_encoder attributes the pipeline has (if any) dynamically as well
# BEGIN: based on https://github.com/huggingface/diffusers/blob/9f5ad1db4197d6c2b503dd5fa3ef4dbec12a4f96/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L531
def dynamically_unscale_lora_layers(
    pipe: DiffusionPipeline
    , lora_scale: Optional[float]   = None
):
    if lora_scale is None or not USE_PEFT_BACKEND:
        return

    text_encoder_attributes: List[str] = [ "text_encoder", "text_encoder_2", "text_encoder_3" ]
    for attribute_name in text_encoder_attributes:
        if hasattr(pipe, attribute_name):
            try:
                encoder_attribute: Optional[Any] = getattr(pipe, attribute_name)
                if encoder_attribute is not None:
                    unscale_lora_layers(encoder_attribute, lora_scale)
            except Exception as e:
                logger.error(f"Couldn't unscale LoRA for pipeline attribute {attribute_name}: {e}\n{traceback.format_exc()}")
  
# END: based on https://github.com/huggingface/diffusers/blob/9f5ad1db4197d6c2b503dd5fa3ef4dbec12a4f96/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L531

def get_weighted_text_embeddings_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str = "",
    neg_prompt: str = "",
    pad_last_block: bool = True,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for Stable Diffusion XL, and extends the behavior with NegPiP routing + dual composition.

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * In the positive prompt: tokens with negative weights are removed from the
            positive stream and their absolute weights are sent to the "neg-from-pos" stream.
        * In the negative prompt: tokens with negative weights are removed from the
            negative stream and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition:
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_embeds            = prompt_pre - neg_total
        * final negative_prompt_embeds   = neg_total + neg_total
        (Per-token weighting is applied to encoder hidden states. SDXL concatenates
        hidden states from text_encoder and text_encoder_2 along the channel dim.)

    Tokenization & grouping:
    - Both CLIP tokenizers are used (tokenizer for text_encoder, tokenizer_2 for text_encoder_2).
    - Prompts are parsed with weights, tokenized, and split into groups of 77 tokens
        ([BOS] + 75 tokens + [EOS]). The shorter side (positive/negative) is padded to keep parity.
    - The last block can optionally be padded with EOS (controlled by `pad_last_block`).

    LoRA scale:
    - If `lora_scale` is provided, LoRA layers on available text encoders are (un)scaled
        around the forward pass to keep behavior consistent with diffusers.

    Clip-skip:
    - If `clip_skip` is provided, the hidden state is taken from `-(clip_skip + k)` where
        `k` is the SDXL-specific final layer index (penultimate). Otherwise the default
        SDXL layer choice is used.

    Args:
        pipe (StableDiffusionXLPipeline)
            A diffusers StableDiffusionXLPipeline. Must provide `tokenizer`, `tokenizer_2`,
            `text_encoder`, and `text_encoder_2`.
        prompt (str)
            A prompt string with weights. Example: "a (white:1.2) cat, (fur:-0.5)".
            Negative weights here are routed to the negative stream as "neg-from-pos".
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        pad_last_block (bool)
            If True, pads the tail block to 75 tokens with EOS so each group forms 77
            with BOS/EOS. If False, the last group is left unpadded.
        lora_scale (Optional[float])
            If provided, temporarily scales LoRA layers on the text encoders during embedding
            computation, restoring them afterward.
        clip_skip (Optional[int])
            Number of final encoder layers to skip when selecting hidden states. When set,
            the hidden state is taken earlier than the default SDXL layer.

    Returns:
        prompt_embeds (torch.Tensor)
            The final positive embeddings after NegPiP dual composition. Shape: [1, T*, C_total].
        negative_prompt_embeds (torch.Tensor)
            The final negative embeddings after NegPiP dual composition. Shape: [1, T*, C_total].
        pooled_prompt_embeds (torch.Tensor)
            SDXL pooled embedding derived from `text_encoder_2`. Shape: [B, C_pool] (diffusers format).
        negative_pooled_prompt_embeds (torch.Tensor)
            SDXL pooled embedding for the negative prompt (from `text_encoder_2`).

    Example:
        from diffusers import StableDiffusionXLPipeline
        import torch

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda:0")

        prompt = "a (white:1.2) cat sitting on a chair, (fur:-0.4)"
        neg_prompt = "(blurry:1.0), (artifact:-0.6)"  # '-0.6' is routed to positive

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            get_weighted_text_embeddings_sdxl(
                pipe=pipe,
                prompt=prompt,
                neg_prompt=neg_prompt,
                pad_last_block=True,
                lora_scale=None,
                clip_skip=None
            )

        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(pipe.device).manual_seed(2)
        ).images[0]
    """

    eos = pipe.tokenizer.eos_token_id
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    # --- 1) tokenize and collect weights for both encoders ---
    # encoder 1 (tokenizer)
    p_tok_1, p_w_1 = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    n_tok_1, n_w_1 = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)
    # encoder 2 (tokenizer_2)
    p_tok_2, p_w_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
    n_tok_2, n_w_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)

    # --- 2) pad positive/negative (for encoder 1) to equal length ---
    # (keep parity with original implementation which drives grouping by tokenizer_1)
    pt_len, nt_len = len(p_tok_1), len(n_tok_1)
    pad = abs(pt_len - nt_len)
    if pt_len > nt_len:
        n_tok_1 += [eos] * pad
        n_w_1   += [1.0] * pad
    elif nt_len > pt_len:
        p_tok_1 += [eos] * pad
        p_w_1   += [1.0] * pad

    # Same parity for encoder 2
    pt2_len, nt2_len = len(p_tok_2), len(n_tok_2)
    pad = abs(pt2_len - nt2_len)
    if pt2_len > nt2_len:
        n_tok_2 += [eos] * pad
        n_w_2   += [1.0] * pad
    elif nt2_len > pt2_len:
        p_tok_2 += [eos] * pad
        p_w_2   += [1.0] * pad

    # --- 3) split weights for routing (only tokenizer_1 weights drive per-token scaling) ---
    # positive side
    _, p_pos_w_1, p_absneg_w_1 = _split_pos_neg_weights(p_w_1)  # pos base; absneg: negative-only part from pos
    # negative side
    _, n_pos_w_1, n_pos_from_neg_1 = _split_pos_neg_weights(n_w_1)  # neg base; pos_from_neg: negative's negative routed to pos

    # --- 4) group into 77-token blocks (tokenizer_1 groups define weights application) ---
    p_groups_1, p_posw_groups_1 = group_tokens_and_weights(p_tok_1.copy(), p_pos_w_1.copy(), pad_last_block)
    _,          p_absneg_groups  = group_tokens_and_weights(p_tok_1.copy(), p_absneg_w_1.copy(), pad_last_block)

    n_groups_1, n_posw_groups_1 = group_tokens_and_weights(n_tok_1.copy(), n_pos_w_1.copy(), pad_last_block)
    _,          n_posfromneg_grp = group_tokens_and_weights(n_tok_1.copy(), n_pos_from_neg_1.copy(), pad_last_block)

    # Mirror grouping for tokenizer_2 (token IDs only; weights are still from tokenizer_1 groups)
    p_groups_2, _ = group_tokens_and_weights(p_tok_2.copy(), [1.0]*len(p_tok_2), pad_last_block)
    n_groups_2, _ = group_tokens_and_weights(n_tok_2.copy(), [1.0]*len(n_tok_2), pad_last_block)

    # --- 5) per-block embeddings ---
    embeds_pos_base, embeds_neg_base = [], []
    embeds_neg_from_pos, embeds_pos_from_neg = [], []

    for i in range(len(p_groups_1)):
        # ===== positive side (base and "neg-from-pos") =====
        tok1 = torch.tensor([p_groups_1[i]], dtype=torch.long, device=pipe.device)
        tok2 = torch.tensor([p_groups_2[i]], dtype=torch.long, device=pipe.device)

        # encoder 1
        pe1 = pipe.text_encoder(tok1.to(pipe.device), output_hidden_states=True)
        hs1 = get_prompt_hidden_states_sdxl(pe1, clip_skip=clip_skip)  # [1, T, C1]
        # encoder 2
        pe2 = pipe.text_encoder_2(tok2.to(pipe.device), output_hidden_states=True)
        hs2 = get_prompt_hidden_states_sdxl(pe2, clip_skip=clip_skip)  # [1, T, C2]
        pooled_prompt_embeds = pe2[0]  # keep last seen pooled (as in original)

        # concat hidden states from enc1/enc2 along channel dim
        emb_pos_concat = torch.concat([hs1, hs2], dim=-1).squeeze(0).to(pipe.device)  # [T, C1+C2]

        # apply positive weights (tokenizer_1 groups)
        w_pos = torch.tensor(p_posw_groups_1[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_pos_concat, w_pos)
        embeds_pos_base.append(emb_pos_concat.unsqueeze(0))  # [1, T, C]

        # build "negative-from-positive" (abs negative weights from positive side)
        emb_neg_from_pos = torch.concat([hs1, hs2], dim=-1).squeeze(0).to(pipe.device)
        w_absneg = torch.tensor(p_absneg_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_neg_from_pos, w_absneg)
        embeds_neg_from_pos.append(emb_neg_from_pos.unsqueeze(0))

        # ===== negative side (base and "pos-from-neg") =====
        tok1_n = torch.tensor([n_groups_1[i]], dtype=torch.long, device=pipe.device)
        tok2_n = torch.tensor([n_groups_2[i]], dtype=torch.long, device=pipe.device)

        ne1 = pipe.text_encoder(tok1_n.to(pipe.device), output_hidden_states=True)
        nhs1 = get_prompt_hidden_states_sdxl(ne1, clip_skip=clip_skip)
        ne2 = pipe.text_encoder_2(tok2_n.to(pipe.device), output_hidden_states=True)
        nhs2 = get_prompt_hidden_states_sdxl(ne2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = ne2[0]  # keep last seen negative pooled (as in original)

        emb_neg_concat = torch.concat([nhs1, nhs2], dim=-1).squeeze(0).to(pipe.device)

        # base negative weights (>=0)
        w_neg_base = torch.tensor(n_posw_groups_1[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_neg_concat, w_neg_base)
        embeds_neg_base.append(emb_neg_concat.unsqueeze(0))

        # build "positive-from-negative" (abs negative weights found in negative prompt)
        emb_pos_from_neg = torch.concat([nhs1, nhs2], dim=-1).squeeze(0).to(pipe.device)
        w_pos_from_neg = torch.tensor(n_posfromneg_grp[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_pos_from_neg, w_pos_from_neg)
        embeds_pos_from_neg.append(emb_pos_from_neg.unsqueeze(0))

    # --- 6) concatenate blocks ---
    prompt_base            = torch.cat(embeds_pos_base,     dim=1)  # [B=1, T*, C]
    negative_base          = torch.cat(embeds_neg_base,     dim=1)
    neg_from_pos_embeds    = torch.cat(embeds_neg_from_pos, dim=1)
    pos_from_neg_embeds    = torch.cat(embeds_pos_from_neg, dim=1)

    # --- 7) merge routed parts ---
    prompt_pre = prompt_base + pos_from_neg_embeds         # positive base plus routed-from-negative
    neg_total  = negative_base + neg_from_pos_embeds       # negative base plus routed-from-positive

    # --- 8) NegPiP dual (alpha=1.0) for token embeddings ---
    prompt_embeds, negative_prompt_embeds = _negpip_dual_apply(prompt_pre, neg_total, alpha=1.0)

    # --- 9) NegPiP dual for pooled embeddings (if available) ---
    pooled_prompt_embeds, negative_pooled_prompt_embeds = _negpip_dual_apply_pooled(
        pooled_prompt_embeds, negative_pooled_prompt_embeds, alpha=1.0
    )

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def get_weighted_text_embeddings_sdxl_refiner(
    pipe: StableDiffusionXLPipeline
    , prompt : str                  = ""
    , neg_prompt: str               = ""
    , lora_scale: Optional[float]   = None
    , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for the SDXL Refiner path and extends the behavior with NegPiP routing + dual composition.

    Refiner specifics:
    - Uses the second encoder only (tokenizer_2 / text_encoder_2), as in the original SDXL Refiner.
    - Per-token weighting follows the original Refiner style:
        * Positive base stream (prompt, w >= 0): apply a tanh-like non-linear mapping, then
            method-2 interpolation against the last token vector:
                ow = w - 1
                tanh_weight = (exp(ow) / (exp(ow) + 1) - 0.5) * 2
                w' = 1 + tanh_weight
                emb[j] = base + (emb[j] - base) * w'
        * Negative base stream (negative prompt, w >= 0): NO tanh, method-2 interpolation only.
        * Routed streams (neg-from-pos, pos-from-neg): NO tanh, method-2 interpolation only.

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * In the positive prompt: tokens with negative weights are removed from the positive stream
            and their absolute weights are sent to the "neg-from-pos" stream.
        * In the negative prompt: tokens with negative weights are removed from the negative stream
            and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition:
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_embeds            = prompt_pre - neg_total
        * final negative_prompt_embeds   = neg_total + neg_total

    Tokenization & grouping:
    - Prompts are parsed with weights using tokenizer_2, then split into 77-token blocks
        ([BOS] + 75 tokens + [EOS]). The shorter side (positive/negative) is padded with EOS
        (Refiner typically uses EOS=49407) to keep parity.

    LoRA scale:
    - If `lora_scale` is provided, LoRA layers on text_encoder_2 are (un)scaled around the forward pass,
        mirroring diffusers' behavior.

    Clip-skip:
    - If `clip_skip` is provided, the hidden state is taken from an earlier layer relative to SDXL's
        default (penultimate) selection.

    Args:
        pipe (StableDiffusionXLPipeline)
            A diffusers SDXL pipeline. Must provide `tokenizer_2` and `text_encoder_2`.
        prompt (str)
            A prompt string with weights. Example: "a (white:1.2) cat, (fur:-0.5)".
            Negative weights here are routed to the negative stream as "neg-from-pos".
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        lora_scale (Optional[float])
            If provided, temporarily scales LoRA layers on `text_encoder_2` during embedding computation,
            restoring them afterward.
        clip_skip (Optional[int])
            Number of final encoder layers to skip when selecting hidden states.

    Returns:
        prompt_embeds (torch.Tensor)
            The final positive embeddings after NegPiP dual composition (Refiner path, encoder_2 only).
            Shape: [1, T*, C2].
        negative_prompt_embeds (torch.Tensor)
            The final negative embeddings after NegPiP dual composition (encoder_2 only).
            Shape: [1, T*, C2].
        pooled_prompt_embeds (torch.Tensor)
            Pooled embedding derived from `text_encoder_2` for the positive prompt (diffusers format).
        negative_pooled_prompt_embeds (torch.Tensor)
            Pooled embedding derived from `text_encoder_2` for the negative prompt.

    Example:
        from diffusers import StableDiffusionXLPipeline
        import torch

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16
        ).to("cuda:0")

        prompt = "a (white:1.2) cat, cinematic lighting, (fur:-0.4)"
        neg_prompt = "(blurry:1.0), (artifact:-0.6)"  # '-0.6' is routed to positive

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            get_weighted_text_embeddings_sdxl_refiner(
                pipe=pipe,
                prompt=prompt,
                neg_prompt=neg_prompt,
                lora_scale=None,
                clip_skip=None
            )

        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(pipe.device).manual_seed(2)
        ).images[0]
    """
    eos = 49407  # refiner path uses constant in the original code
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    # 1) tokenize via tokenizer_2 only
    p_tok2, p_w2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
    n_tok2, n_w2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)

    # 2) pad parity
    lp, ln = len(p_tok2), len(n_tok2)
    pad = abs(lp - ln)
    if lp > ln:
        n_tok2 += [eos] * pad
        n_w2   += [1.0] * pad
    elif ln > lp:
        p_tok2 += [eos] * pad
        p_w2   += [1.0] * pad

    # 3) split weights for routing
    _, p_pos_w2, p_absneg_w2  = _split_pos_neg_weights(p_w2)   # positive base (>=0), neg-from-pos (abs)
    _, n_pos_w2, n_posfrom_w2 = _split_pos_neg_weights(n_w2)   # negative base (>=0), pos-from-neg (abs)

    # 4) group (77-token blocks)
    p_groups_2, p_posw_groups_2   = group_tokens_and_weights(p_tok2.copy(), p_pos_w2.copy())
    _,           p_absneg_groups  = group_tokens_and_weights(p_tok2.copy(), p_absneg_w2.copy())
    n_groups_2, n_posw_groups_2   = group_tokens_and_weights(n_tok2.copy(), n_pos_w2.copy())
    _,           n_posfrom_groups = group_tokens_and_weights(n_tok2.copy(), n_posfrom_w2.copy())

    # 5) embed per block
    embeds_pos_base, embeds_neg_base = [], []
    embeds_neg_from_pos, embeds_pos_from_neg = [], []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    for i in range(len(p_groups_2)):
        # ----- positive side (BASE with tanh) -----
        tok2 = torch.tensor([p_groups_2[i]], dtype=torch.long, device=pipe.device)
        pe2  = pipe.text_encoder_2(tok2.to(pipe.device), output_hidden_states=True)
        hs2  = get_prompt_hidden_states_sdxl(pe2, clip_skip=clip_skip)   # [1,T,C]
        pooled_prompt_embeds = pe2[0]

        emb_pos = hs2.squeeze(0).to(pipe.device)
        w_p     = torch.tensor(p_posw_groups_2[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_tanh_inplace(emb_pos, w_p)                         # tanh + method-2
        embeds_pos_base.append(emb_pos.unsqueeze(0))

        # neg-from-pos (NO tanh; linear method-2)
        emb_nfp = hs2.squeeze(0).to(pipe.device)
        w_nfp   = torch.tensor(p_absneg_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb_nfp, w_nfp)                     # linear method-2
        embeds_neg_from_pos.append(emb_nfp.unsqueeze(0))

        # ----- negative side (BASE without tanh) -----
        tok2n = torch.tensor([n_groups_2[i]], dtype=torch.long, device=pipe.device)
        ne2   = pipe.text_encoder_2(tok2n.to(pipe.device), output_hidden_states=True)
        nhs2  = get_prompt_hidden_states_sdxl(ne2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = ne2[0]

        emb_neg = nhs2.squeeze(0).to(pipe.device)
        w_n     = torch.tensor(n_posw_groups_2[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb_neg, w_n)                        # linear method-2 (no tanh)
        embeds_neg_base.append(emb_neg.unsqueeze(0))

        # pos-from-neg (NO tanh; linear method-2)
        emb_pfn = nhs2.squeeze(0).to(pipe.device)
        w_pfn   = torch.tensor(n_posfrom_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb_pfn, w_pfn)                      # linear method-2
        embeds_pos_from_neg.append(emb_pfn.unsqueeze(0))

    # 6) concat blocks
    prompt_base         = torch.cat(embeds_pos_base,     dim=1)
    negative_base       = torch.cat(embeds_neg_base,     dim=1)
    neg_from_pos_embeds = torch.cat(embeds_neg_from_pos, dim=1)
    pos_from_neg_embeds = torch.cat(embeds_pos_from_neg, dim=1)

    # 7) merge routed parts
    prompt_pre = prompt_base + pos_from_neg_embeds
    neg_total  = negative_base + neg_from_pos_embeds

    # 8) NegPiP dual (alpha=1.0)
    prompt_embeds, negative_prompt_embeds = _negpip_dual_apply(prompt_pre, neg_total, alpha=1.0)

    # 9) pooled also gets NegPiP dual
    pooled_prompt_embeds, negative_pooled_prompt_embeds = _negpip_dual_apply_pooled(
        pooled_prompt_embeds, negative_pooled_prompt_embeds, alpha=1.0
    )

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def get_weighted_text_embeddings_sdxl_2p(
    pipe: StableDiffusionXLPipeline
    , prompt : str                  = ""
    , prompt_2 : str                = None
    , neg_prompt: str               = ""
    , neg_prompt_2: str             = None
    , lora_scale: Optional[float]   = None
    , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for Stable Diffusion XL (two-prompt, "2p" path), and extends the behavior with
    NegPiP routing + dual composition.

    2p specifics:
    - Uses both SDXL text encoders:
        * encoder #1: tokenizer / text_encoder
        * encoder #2: tokenizer_2 / text_encoder_2
    - Hidden states from both encoders are concatenated along the channel dimension.
    - Per-token weighting in 2p uses method-2 interpolation (linear) only:
            emb[j] = base + (emb[j] - base) * w
        (No tanh non-linearity is applied in 2p.)

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * Positive prompt:  tokens with negative weights are removed from the positive stream
            and their absolute weights are sent to the "neg-from-pos" stream.
        * Negative prompt:  tokens with negative weights are removed from the negative stream
            and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition:
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_embeds            = prompt_pre - neg_total
        * final negative_prompt_embeds   = neg_total + neg_total

    Tokenization & grouping:
    - Each prompt is parsed with weights, tokenized by both tokenizers, and split into groups of
        77 tokens ([BOS] + 75 tokens + [EOS]). The shorter side (positive/negative) is padded with EOS
        to keep parity. The last block may be padded depending on the implementation choice.

    LoRA scale:
    - If `lora_scale` is provided, LoRA layers on the available text encoders are (un)scaled
        around the forward pass to mirror diffusers behavior.

    Clip-skip:
    - If `clip_skip` is provided, the hidden state is selected from an earlier layer than SDXL's
        default (penultimate) selection.

    Args:
        pipe (StableDiffusionXLPipeline)
            A diffusers SDXL pipeline. Must provide `tokenizer`, `tokenizer_2`,
            `text_encoder`, and `text_encoder_2`.
        prompt (str)
            A prompt string with weights for encoder #1 / #2 (paired with tokenizer & tokenizer_2).
            Example: "a (white:1.2) cat, (fur:-0.5)". Negative weights here are routed to the negative
            stream as "neg-from-pos".
        prompt_2 (str)
            An alternative prompt string with weights used specifically for `tokenizer_2` /
            `text_encoder_2`. If None, it falls back to `prompt`.
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        neg_prompt_2 (str)
            An alternative negative prompt string with weights for `tokenizer_2` /
            `text_encoder_2`. If None, it falls back to `neg_prompt`.
        lora_scale (Optional[float])
            If provided, temporarily scales LoRA layers on both text encoders during embedding
            computation, restoring them afterward.
        clip_skip (Optional[int])
            Number of final encoder layers to skip when selecting hidden states.

    Returns:
        prompt_embeds (torch.Tensor)
            The final positive embeddings after NegPiP dual composition.
            Shape: [1, T*, C_total] where C_total = C_enc1 + C_enc2.
        negative_prompt_embeds (torch.Tensor)
            The final negative embeddings after NegPiP dual composition.
            Shape: [1, T*, C_total].
        pooled_prompt_embeds (torch.Tensor)
            SDXL pooled embedding derived from `text_encoder_2` for the positive prompt (diffusers format).
        negative_pooled_prompt_embeds (torch.Tensor)
            SDXL pooled embedding derived from `text_encoder_2` for the negative prompt.

    Example:
        from diffusers import StableDiffusionXLPipeline
        import torch

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda:0")

        prompt     = "a (white:1.2) cat on a chair, (fur:-0.4)"
        prompt_2   = "a (white:1.1) cat on a chair, cozy room"
        neg_prompt = "(blurry:1.0), (artifact:-0.6)"
        neg_prompt_2 = "(jpeg:1.0)"

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            get_weighted_text_embeddings_sdxl_2p(
                pipe=pipe,
                prompt=prompt,
                prompt_2=prompt_2,
                neg_prompt=neg_prompt,
                neg_prompt_2=neg_prompt_2,
                lora_scale=None,
                clip_skip=None
            )

        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(pipe.device).manual_seed(2)
        ).images[0]
    """
    prompt_2     = prompt_2 or prompt
    neg_prompt_2 = neg_prompt_2 or neg_prompt

    eos = pipe.tokenizer.eos_token_id
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    # 1) tokenize for both encoders
    # encoder 1 (tokenizer)
    p1_tok, p1_w = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    n1_tok, n1_w = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)
    # encoder 2 (tokenizer_2)
    p2_tok, p2_w = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt_2)
    n2_tok, n2_w = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt_2)

    # 2) pad parity within each encoder (ids and weights)
    def _pad_pair(tok_a, w_a, tok_b, w_b, eos_id):
        la, lb = len(tok_a), len(tok_b)
        if la > lb:
            pad = la - lb
            tok_b += [eos_id] * pad
            w_b   += [1.0] * pad
        elif lb > la:
            pad = lb - la
            tok_a += [eos_id] * pad
            w_a   += [1.0] * pad
        return tok_a, w_a, tok_b, w_b

    p1_tok, p1_w, n1_tok, n1_w = _pad_pair(p1_tok, p1_w, n1_tok, n1_w, eos)
    p2_tok, p2_w, n2_tok, n2_w = _pad_pair(p2_tok, p2_w, n2_tok, n2_w, eos)

    # 3) split weights per encoder (routing)
    # enc1
    _, p1_pos_w, p1_absneg_w = _split_pos_neg_weights(p1_w)  # positive base (>=0), neg-from-pos (abs)
    _, n1_pos_w, n1_posfrom  = _split_pos_neg_weights(n1_w)  # negative base (>=0), pos-from-neg (abs)
    # enc2
    _, p2_pos_w, p2_absneg_w = _split_pos_neg_weights(p2_w)
    _, n2_pos_w, n2_posfrom  = _split_pos_neg_weights(n2_w)

    # 4) group per encoder into 77-token blocks
    p1_groups, p1_pos_groups   = group_tokens_and_weights(p1_tok.copy(), p1_pos_w.copy())
    _,  p1_absneg_groups       = group_tokens_and_weights(p1_tok.copy(), p1_absneg_w.copy())
    n1_groups, n1_pos_groups   = group_tokens_and_weights(n1_tok.copy(), n1_pos_w.copy())
    _,  n1_posfrom_groups      = group_tokens_and_weights(n1_tok.copy(), n1_posfrom.copy())

    p2_groups, p2_pos_groups   = group_tokens_and_weights(p2_tok.copy(), p2_pos_w.copy())
    _,  p2_absneg_groups       = group_tokens_and_weights(p2_tok.copy(), p2_absneg_w.copy())
    n2_groups, n2_pos_groups   = group_tokens_and_weights(n2_tok.copy(), n2_pos_w.copy())
    _,  n2_posfrom_groups      = group_tokens_and_weights(n2_tok.copy(), n2_posfrom.copy())

    # 5) per-block embeddings (enc1 + enc2; method-2 linear for all)
    embeds_pos_base, embeds_neg_base = [], []
    embeds_neg_from_pos, embeds_pos_from_neg = [], []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    for i in range(len(p1_groups)):
        # ----- positive side -----
        # enc1
        t1 = torch.tensor([p1_groups[i]], dtype=torch.long, device=pipe.device)
        pe1 = pipe.text_encoder(t1.to(pipe.device), output_hidden_states=True)
        hs1 = get_prompt_hidden_states_sdxl(pe1, clip_skip=clip_skip)  # [1,T,C1]
        emb1 = hs1.squeeze(0).to(pipe.device)
        w1   = torch.tensor(p1_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb1, w1)

        # enc2
        t2 = torch.tensor([p2_groups[i]], dtype=torch.long, device=pipe.device)
        pe2 = pipe.text_encoder_2(t2.to(pipe.device), output_hidden_states=True)
        hs2 = get_prompt_hidden_states_sdxl(pe2, clip_skip=clip_skip)  # [1,T,C2]
        pooled_prompt_embeds = pe2[0]  # keep latest pooled, as in original
        emb2 = hs2.squeeze(0).to(pipe.device)
        w2   = torch.tensor(p2_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb2, w2)

        emb_pos_concat = torch.cat([emb1, emb2], dim=-1).unsqueeze(0)  # [1,T,C1+C2]
        embeds_pos_base.append(emb_pos_concat)

        # negative-from-positive (routing) — linear method-2
        # enc1
        emb1_nfp = hs1.squeeze(0).to(pipe.device)
        w1_nfp   = torch.tensor(p1_absneg_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb1_nfp, w1_nfp)
        # enc2
        emb2_nfp = hs2.squeeze(0).to(pipe.device)
        w2_nfp   = torch.tensor(p2_absneg_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(emb2_nfp, w2_nfp)

        emb_nfp_concat = torch.cat([emb1_nfp, emb2_nfp], dim=-1).unsqueeze(0)
        embeds_neg_from_pos.append(emb_nfp_concat)

        # ----- negative side -----
        # enc1
        tn1 = torch.tensor([n1_groups[i]], dtype=torch.long, device=pipe.device)
        ne1 = pipe.text_encoder(tn1.to(pipe.device), output_hidden_states=True)
        nhs1 = get_prompt_hidden_states_sdxl(ne1, clip_skip=clip_skip)
        nemb1 = nhs1.squeeze(0).to(pipe.device)
        nw1   = torch.tensor(n1_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(nemb1, nw1)

        # enc2
        tn2 = torch.tensor([n2_groups[i]], dtype=torch.long, device=pipe.device)
        ne2 = pipe.text_encoder_2(tn2.to(pipe.device), output_hidden_states=True)
        nhs2 = get_prompt_hidden_states_sdxl(ne2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = ne2[0]
        nemb2 = nhs2.squeeze(0).to(pipe.device)
        nw2   = torch.tensor(n2_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(nemb2, nw2)

        emb_neg_concat = torch.cat([nemb1, nemb2], dim=-1).unsqueeze(0)
        embeds_neg_base.append(emb_neg_concat)

        # positive-from-negative (routing) — linear method-2
        pfn1 = nhs1.squeeze(0).to(pipe.device)
        w_pfn1 = torch.tensor(n1_posfrom_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(pfn1, w_pfn1)

        pfn2 = nhs2.squeeze(0).to(pipe.device)
        w_pfn2 = torch.tensor(n2_posfrom_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_method2_linear_inplace(pfn2, w_pfn2)

        emb_pfn_concat = torch.cat([pfn1, pfn2], dim=-1).unsqueeze(0)
        embeds_pos_from_neg.append(emb_pfn_concat)

    # 6) concat across sequence
    prompt_base         = torch.cat(embeds_pos_base,     dim=1)  # [1, T*, C]
    negative_base       = torch.cat(embeds_neg_base,     dim=1)
    neg_from_pos_embeds = torch.cat(embeds_neg_from_pos, dim=1)
    pos_from_neg_embeds = torch.cat(embeds_pos_from_neg, dim=1)

    # 7) merge routed parts
    prompt_pre = prompt_base + pos_from_neg_embeds
    neg_total  = negative_base + neg_from_pos_embeds

    # 8) NegPiP dual (alpha=1.0)
    prompt_embeds, negative_prompt_embeds = _negpip_dual_apply(prompt_pre, neg_total, alpha=1.0)

    # 9) pooled also gets NegPiP dual
    pooled_prompt_embeds, negative_pooled_prompt_embeds = _negpip_dual_apply_pooled(
        pooled_prompt_embeds, negative_pooled_prompt_embeds, alpha=1.0
    )

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_s_cascade(
        pipe: typing.Union[StableCascadePriorPipeline, StableCascadeDecoderPipeline]
        , prompt: str                   = ""
        , neg_prompt: str               = ""
        , pad_last_block: bool          = True
        , lora_scale: Optional[float]   = None
        , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for Stable Cascade, and extends the behavior with NegPiP routing + dual composition.

    Cascade specifics:
    - Uses a single tokenizer / text_encoder (the Cascade text encoder).
    - Per-token weighting is applied linearly to hidden states (no tanh for Cascade).
    - Hidden states are taken from the Cascade-specific final layer index the same way as the
        original implementation (via get_prompt_hidden_states_s_cascade / optional clip_skip).

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * Positive prompt: tokens with negative weights are removed from the positive stream
            and their absolute weights are sent to the "neg-from-pos" stream.
        * Negative prompt: tokens with negative weights are removed from the negative stream
            and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition:
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_embeds            = prompt_pre - neg_total
        * final negative_prompt_embeds   = neg_total + neg_total

    Tokenization & grouping:
    - The prompt and negative prompt are parsed with weights, tokenized, and split into
        groups of 77 tokens ([BOS] + 75 tokens + [EOS]).
    - To keep parity between positive and negative sides, the shorter sequence is padded
        with EOS and weight 1.0. The last block can be padded with EOS depending on
        `pad_last_block`.

    LoRA scale:
    - If `lora_scale` is provided, LoRA layers on the text encoder are (un)scaled around
        the forward pass to mirror diffusers behavior.

    Clip-skip:
    - If `clip_skip` is provided, the hidden state is taken from an earlier layer relative
        to the Cascade default selected layer.

    Args:
        pipe (typing.Union[StableCascadePriorPipeline, StableCascadeDecoderPipeline])
            A Stable Cascade pipeline (prior or decoder). Must provide `tokenizer` and `text_encoder`.
        prompt (str)
            A prompt string with weights. Example: "a (white:1.2) cat, (fur:-0.5)".
            Negative weights here are routed to the negative stream as "neg-from-pos".
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        pad_last_block (bool)
            If True, pads the final block to 75 tokens (so each group forms [BOS]+75+[EOS]).
            If False, the final block is left unpadded.
        lora_scale (Optional[float])
            If provided, temporarily scales LoRA layers on the text encoder during embedding
            computation, restoring them afterward.
        clip_skip (Optional[int])
            Number of final encoder layers to skip when selecting hidden states.

    Returns:
        prompt_embeds (torch.Tensor)
            The final positive token embeddings after NegPiP dual composition. Shape: [1, T*, C].
        negative_prompt_embeds (torch.Tensor)
            The final negative token embeddings after NegPiP dual composition. Shape: [1, T*, C].
        pooled_prompt_embeds (torch.Tensor)
            Cascade pooled embedding for the positive prompt (diffusers format, typically
            `text_embeds.unsqueeze(1)` from the text encoder).
        negative_pooled_prompt_embeds (torch.Tensor)
            Cascade pooled embedding for the negative prompt.

    Example:
        # Prior or Decoder pipeline (use the one appropriate for your stage)
        # from diffusers import StableCascadePriorPipeline
        # pipe = StableCascadePriorPipeline.from_pretrained("...", torch_dtype=torch.float16).to("cuda:0")

        # from diffusers import StableCascadeDecoderPipeline
        # pipe = StableCascadeDecoderPipeline.from_pretrained("...", torch_dtype=torch.float16).to("cuda:0")

        import torch

        prompt = "a (white:1.2) cat, high detail, (fur:-0.4)"
        neg_prompt = "(blurry:1.0), (artifact:-0.6)"  # '-0.6' is routed to positive

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            get_weighted_text_embeddings_s_cascade(
                pipe=pipe,
                prompt=prompt,
                neg_prompt=neg_prompt,
                pad_last_block=True,
                lora_scale=None,
                clip_skip=None
            )

        # Use the returned embeddings with your Cascade pipeline call, e.g.:
        # images = pipe(
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #     generator=torch.Generator(pipe.device).manual_seed(2)
        # ).images
    """
    eos = pipe.tokenizer.eos_token_id

    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    # 1) tokenize (single encoder for Cascade)
    p_tok, p_w = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    n_tok, n_w = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

    # 2) pad parity between positive/negative
    lp, ln = len(p_tok), len(n_tok)
    pad = abs(lp - ln)
    if lp > ln:
        n_tok += [eos] * pad
        n_w   += [1.0] * pad
    elif ln > lp:
        p_tok += [eos] * pad
        p_w   += [1.0] * pad

    # 3) split weights for routing (linear scaling only)
    #    positive side -> base (>=0), neg-from-pos (abs of negative weights)
    _, p_pos_w, p_absneg_w = _split_pos_neg_weights(p_w)
    #    negative side -> base (>=0), pos-from-neg (abs of negative weights)
    _, n_pos_w, n_posfrom_w = _split_pos_neg_weights(n_w)

    # 4) group into 77-token blocks
    p_groups, p_pos_groups   = group_tokens_and_weights(p_tok.copy(), p_pos_w.copy(), pad_last_block=pad_last_block)
    _,        p_absneg_groups= group_tokens_and_weights(p_tok.copy(), p_absneg_w.copy(), pad_last_block=pad_last_block)

    n_groups, n_pos_groups   = group_tokens_and_weights(n_tok.copy(), n_pos_w.copy(), pad_last_block=pad_last_block)
    _,        n_posfrom_groups= group_tokens_and_weights(n_tok.copy(), n_posfrom_w.copy(), pad_last_block=pad_last_block)

    # 5) per-block embeddings (single text_encoder; keep VRAM usage similar to original: move intermediates to CPU)
    embeds_pos_base, embeds_neg_base = [], []
    embeds_neg_from_pos, embeds_pos_from_neg = [], []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    for i in range(len(p_groups)):
        # ----- positive side -----
        tok = torch.tensor([p_groups[i]], dtype=torch.long, device=pipe.device)
        pe  = pipe.text_encoder(tok.to(pipe.device), output_hidden_states=True)
        hs  = get_prompt_hidden_states_s_cascade(pe, clip_skip=clip_skip)  # [1, T, C]
        pooled_prompt_embeds = pe.text_embeds.unsqueeze(1)

        # base positive (>=0 weights) — linear multiplication
        emb_pos = hs.squeeze(0).to(pipe.device)
        w_pos   = torch.tensor(p_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_pos, w_pos)
        embeds_pos_base.append(emb_pos.unsqueeze(0).cpu())

        # neg-from-pos (abs negative weights from positive) — linear multiplication
        emb_nfp = hs.squeeze(0).to(pipe.device)
        w_nfp   = torch.tensor(p_absneg_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_nfp, w_nfp)
        embeds_neg_from_pos.append(emb_nfp.unsqueeze(0).cpu())

        # ----- negative side -----
        tok_n = torch.tensor([n_groups[i]], dtype=torch.long, device=pipe.device)
        ne    = pipe.text_encoder(tok_n.to(pipe.device), output_hidden_states=True)
        nhs   = get_prompt_hidden_states_s_cascade(ne, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = ne.text_embeds.unsqueeze(1)

        # base negative (>=0 weights) — linear multiplication
        emb_neg = nhs.squeeze(0).to(pipe.device)
        w_neg   = torch.tensor(n_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_neg, w_neg)
        embeds_neg_base.append(emb_neg.unsqueeze(0).cpu())

        # pos-from-neg (abs negative weights from negative) — linear multiplication
        emb_pfn = nhs.squeeze(0).to(pipe.device)
        w_pfn   = torch.tensor(n_posfrom_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb_pfn, w_pfn)
        embeds_pos_from_neg.append(emb_pfn.unsqueeze(0).cpu())

    # 6) concatenate blocks (move back to device)
    prompt_base         = torch.cat(embeds_pos_base,     dim=1).to(pipe.device)
    negative_base       = torch.cat(embeds_neg_base,     dim=1).to(pipe.device)
    neg_from_pos_embeds = torch.cat(embeds_neg_from_pos, dim=1).to(pipe.device)
    pos_from_neg_embeds = torch.cat(embeds_pos_from_neg, dim=1).to(pipe.device)

    # 7) merge routed parts
    prompt_pre = prompt_base + pos_from_neg_embeds
    neg_total  = negative_base + neg_from_pos_embeds

    # 8) NegPiP dual (alpha=1.0) for token embeddings
    prompt_embeds, negative_prompt_embeds = _negpip_dual_apply(prompt_pre, neg_total, alpha=1.0)

    # 9) NegPiP dual for pooled embeddings
    pooled_prompt_embeds, negative_pooled_prompt_embeds = _negpip_dual_apply_pooled(
        pooled_prompt_embeds, negative_pooled_prompt_embeds, alpha=1.0
    )

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_sd3(
    pipe: StableDiffusion3Pipeline
    , prompt : str                  = ""
    , neg_prompt: str               = ""
    , pad_last_block                = True
    , use_t5_encoder                = True
    , lora_scale: Optional[float]   = None
    , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompts with per-token weights (no length limitation)
    for Stable Diffusion 3, and extends the behavior with NegPiP routing + dual composition.

    SD3 encoders:
    - encoder #1: tokenizer / text_encoder          (CLIP A)
    - encoder #2: tokenizer_2 / text_encoder_2      (CLIP B)
    - encoder #3: tokenizer_3 / text_encoder_3 (T5) (optional; enabled via `use_t5_encoder`)

    Weighting policy:
    - CLIP A / CLIP B: per-token weights are applied linearly to hidden states (no tanh).
    - T5: per-token weights are also applied linearly to token embeddings.
    - Grouping for CLIP paths uses 77-token blocks ([BOS] + 75 + [EOS]); T5 is processed
        for the full sequence (no 77-block split).

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * Positive prompt:  tokens with negative weights are removed from the positive stream,
            and their absolute weights are sent to the "neg-from-pos" stream.
        * Negative prompt:  tokens with negative weights are removed from the negative stream,
            and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition (per-encoder stream, then merged):
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final prompt_stream        = prompt_pre - neg_total
        * final negative_stream      = neg_total + neg_total
    - After CLIP-side NegPiP, CLIP streams (A+B) are merged as in the original SD3.

    CLIP+T5 merge:
    - As in the original SD3, CLIP (A+B) token embeddings are padded on the last channel to match
        T5's last dimension, then concatenated with T5 along the sequence dimension (-2).

    LoRA scale:
    - If `lora_scale` is provided, LoRA layers on available text encoders are (un)scaled around
        the forward pass to mirror diffusers behavior.

    Clip-skip:
    - If `clip_skip` is provided, the hidden state is taken from an earlier layer relative to SD3's
        default (same index policy as SDXL).

    Args:
        pipe (StableDiffusion3Pipeline)
            A diffusers SD3 pipeline. Must provide `tokenizer`, `tokenizer_2`, `tokenizer_3`
            and corresponding `text_encoder` modules.
        prompt (str)
            A prompt string with weights. Example: "a (white:1.2) cat, (fur:-0.5)".
            Negative weights here are routed to the negative stream as "neg-from-pos".
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        pad_last_block (bool)
            If True, pads the tail block for CLIP token streams to form [BOS]+75+[EOS] per group.
            If False, the final CLIP block is left unpadded. (T5 is unaffected.)
        use_t5_encoder (bool)
            If True, uses encoder #3 (T5) and applies linear per-token scaling on T5 embeddings.
            If False, a zero tensor is used in place of T5 embeddings (shape-compatible).
        lora_scale (Optional[float])
            If provided, temporarily scales LoRA layers on the text encoders during embedding
            computation, restoring them afterward.
        clip_skip (Optional[int])
            Number of final encoder layers to skip when selecting hidden states (CLIP A/B).

    Returns:
        sd3_prompt_embeds (torch.Tensor)
            Final positive embeddings after NegPiP dual and CLIP+T5 merge.
            Shape: [1, T_total, C_t5], where CLIP channels are padded to T5 width.
        sd3_neg_prompt_embeds (torch.Tensor)
            Final negative embeddings after NegPiP dual and CLIP+T5 merge.
            Shape: [1, T_total, C_t5].
        pooled_prompt_embeds (torch.Tensor)
            Pooled embeddings for the positive prompt from CLIP A and CLIP B, concatenated on
            the channel dimension (diffusers format).
        negative_pooled_prompt_embeds (torch.Tensor)
            Pooled embeddings for the negative prompt from CLIP A and CLIP B, concatenated.

    Example:
        from diffusers import StableDiffusion3Pipeline
        import torch

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3",
            torch_dtype=torch.float16
        ).to("cuda:0")

        prompt     = "a (white:1.2) cat on a chair, (fur:-0.4)"
        neg_prompt = "(blurry:1.0), (artifact:-0.6)"  # '-0.6' is routed to positive

        sd3_prompt_embeds, sd3_neg_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            get_weighted_text_embeddings_sd3(
                pipe=pipe,
                prompt=prompt,
                neg_prompt=neg_prompt,
                pad_last_block=True,
                use_t5_encoder=True,
                lora_scale=None,
                clip_skip=None
            )

        image = pipe(
            prompt_embeds=sd3_prompt_embeds,
            negative_prompt_embeds=sd3_neg_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(pipe.device).manual_seed(2)
        ).images[0]
    """
    eos = pipe.tokenizer.eos_token_id
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    # --- 1) tokenize & weights for all encoders ---
    # CLIP A
    p1_tok, p1_w = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    n1_tok, n1_w = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)
    # CLIP B
    p2_tok, p2_w = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
    n2_tok, n2_w = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)
    # T5
    p3_tok, p3_w = get_prompts_tokens_with_weights_t5(pipe.tokenizer_3, prompt)
    n3_tok, n3_w = get_prompts_tokens_with_weights_t5(pipe.tokenizer_3, neg_prompt)

    # --- 2) pad parity within CLIP encoders (77-token grouping depends on parity) ---
    def _pad_pair(tok_a, w_a, tok_b, w_b, eos_id):
        la, lb = len(tok_a), len(tok_b)
        pad = abs(la - lb)
        if la > lb:
            tok_b += [eos_id] * pad
            w_b   += [1.0] * pad
        elif lb > la:
            tok_a += [eos_id] * pad
            w_a   += [1.0] * pad
        return tok_a, w_a, tok_b, w_b

    p1_tok, p1_w, n1_tok, n1_w = _pad_pair(p1_tok, p1_w, n1_tok, n1_w, eos)
    p2_tok, p2_w, n2_tok, n2_w = _pad_pair(p2_tok, p2_w, n2_tok, n2_w, eos)

    # --- 3) split weights per encoder (routing) ---
    # CLIP A
    _, p1_pos_w, p1_absneg_w = _split_pos_neg_weights(p1_w)  # pos base, neg-from-pos
    _, n1_pos_w, n1_posfrom  = _split_pos_neg_weights(n1_w)  # neg base, pos-from-neg
    # CLIP B
    _, p2_pos_w, p2_absneg_w = _split_pos_neg_weights(p2_w)
    _, n2_pos_w, n2_posfrom  = _split_pos_neg_weights(n2_w)
    # T5
    _, p3_pos_w, p3_absneg_w = _split_pos_neg_weights(p3_w)
    _, n3_pos_w, n3_posfrom  = _split_pos_neg_weights(n3_w)

    # --- 4) group CLIP tokens into 77-token blocks (A/B) ---
    p1_groups, p1_pos_groups = group_tokens_and_weights(p1_tok.copy(), p1_pos_w.copy(), pad_last_block=pad_last_block)
    _,         p1_abs_groups = group_tokens_and_weights(p1_tok.copy(), p1_absneg_w.copy(), pad_last_block=pad_last_block)
    n1_groups, n1_pos_groups = group_tokens_and_weights(n1_tok.copy(), n1_pos_w.copy(), pad_last_block=pad_last_block)
    _,         n1_pfn_groups = group_tokens_and_weights(n1_tok.copy(), n1_posfrom.copy(),  pad_last_block=pad_last_block)

    p2_groups, p2_pos_groups = group_tokens_and_weights(p2_tok.copy(), p2_pos_w.copy(), pad_last_block=pad_last_block)
    _,         p2_abs_groups = group_tokens_and_weights(p2_tok.copy(), p2_absneg_w.copy(), pad_last_block=pad_last_block)
    n2_groups, n2_pos_groups = group_tokens_and_weights(n2_tok.copy(), n2_pos_w.copy(), pad_last_block=pad_last_block)
    _,         n2_pfn_groups = group_tokens_and_weights(n2_tok.copy(), n2_posfrom.copy(),  pad_last_block=pad_last_block)

    # --- 5) per-block embeddings for CLIP A/B (linear scaling only) ---
    pos_base_blocks, neg_base_blocks = [], []
    neg_from_pos_blocks, pos_from_neg_blocks = [], []
    pooled_prompt_embeds_1 = None
    pooled_prompt_embeds_2 = None
    negative_pooled_prompt_embeds_1 = None
    negative_pooled_prompt_embeds_2 = None

    for i in range(len(p1_groups)):
        # ----- POSITIVE -----
        t1 = torch.tensor([p1_groups[i]], dtype=torch.long, device=pipe.device)
        e1 = pipe.text_encoder(t1.to(pipe.device), output_hidden_states=True)
        hs1 = get_prompt_hidden_states_sd3(e1, clip_skip=clip_skip)   # [1,T,C1]
        pooled_prompt_embeds_1 = e1[0]

        t2 = torch.tensor([p2_groups[i]], dtype=torch.long, device=pipe.device)
        e2 = pipe.text_encoder_2(t2.to(pipe.device), output_hidden_states=True)
        hs2 = get_prompt_hidden_states_sd3(e2, clip_skip=clip_skip)   # [1,T,C2]
        pooled_prompt_embeds_2 = e2[0]

        # base positive (>=0)
        emb1 = hs1.squeeze(0).to(pipe.device); w1 = torch.tensor(p1_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb1, w1)
        emb2 = hs2.squeeze(0).to(pipe.device); w2 = torch.tensor(p2_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb2, w2)
        pos_base_blocks.append(torch.cat([emb1, emb2], dim=-1).unsqueeze(0))

        # neg-from-pos (abs negative from positive)
        emb1_nfp = hs1.squeeze(0).to(pipe.device); w1_nfp = torch.tensor(p1_abs_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb1_nfp, w1_nfp)
        emb2_nfp = hs2.squeeze(0).to(pipe.device); w2_nfp = torch.tensor(p2_abs_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(emb2_nfp, w2_nfp)
        neg_from_pos_blocks.append(torch.cat([emb1_nfp, emb2_nfp], dim=-1).unsqueeze(0))

        # ----- NEGATIVE -----
        tn1 = torch.tensor([n1_groups[i]], dtype=torch.long, device=pipe.device)
        ne1 = pipe.text_encoder(tn1.to(pipe.device), output_hidden_states=True)
        nhs1 = get_prompt_hidden_states_sd3(ne1, clip_skip=clip_skip)
        negative_pooled_prompt_embeds_1 = ne1[0]

        tn2 = torch.tensor([n2_groups[i]], dtype=torch.long, device=pipe.device)
        ne2 = pipe.text_encoder_2(tn2.to(pipe.device), output_hidden_states=True)
        nhs2 = get_prompt_hidden_states_sd3(ne2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds_2 = ne2[0]

        # base negative (>=0)
        nemb1 = nhs1.squeeze(0).to(pipe.device); nw1 = torch.tensor(n1_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(nemb1, nw1)
        nemb2 = nhs2.squeeze(0).to(pipe.device); nw2 = torch.tensor(n2_pos_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(nemb2, nw2)
        neg_base_blocks.append(torch.cat([nemb1, nemb2], dim=-1).unsqueeze(0))

        # pos-from-neg (abs negative from negative)
        pfn1 = nhs1.squeeze(0).to(pipe.device); w_pfn1 = torch.tensor(n1_pfn_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(pfn1, w_pfn1)
        pfn2 = nhs2.squeeze(0).to(pipe.device); w_pfn2 = torch.tensor(n2_pfn_groups[i], dtype=torch.float16, device=pipe.device)
        _apply_token_weights_inplace(pfn2, w_pfn2)
        pos_from_neg_blocks.append(torch.cat([pfn1, pfn2], dim=-1).unsqueeze(0))

    # concat CLIP blocks across sequence
    clip_pos_base         = torch.cat(pos_base_blocks,     dim=1)  # [1, Tclip, Cclip]
    clip_neg_base         = torch.cat(neg_base_blocks,     dim=1)
    clip_neg_from_pos     = torch.cat(neg_from_pos_blocks, dim=1)
    clip_pos_from_neg     = torch.cat(pos_from_neg_blocks, dim=1)

    # merge routed for CLIP
    clip_prompt_pre = clip_pos_base + clip_pos_from_neg
    clip_neg_total  = clip_neg_base + clip_neg_from_pos

    # NegPiP dual on CLIP token embeddings
    clip_prompt_after, clip_neg_after = _negpip_dual_apply(clip_prompt_pre, clip_neg_total, alpha=1.0)

    # pooled embeddings (CLIP) — concatenate enc1/enc2 pooled
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1)

    # --- 6) T5 embeddings (linear scaling, whole sequence; optional) ---
    if use_t5_encoder and pipe.text_encoder_3:
        # prompt side
        p3_ids = torch.tensor([p3_tok], dtype=torch.long)
        t5_p   = pipe.text_encoder_3(p3_ids.to(pipe.device))[0].squeeze(0)  # [T5, Ct5]
        t5_p   = t5_p.to(device=pipe.device)
        # base (>=0)
        t5_pb  = t5_p.clone()
        for j,w in enumerate(p3_pos_w):
            if w != 1.0: t5_pb[j] = t5_pb[j] * w
        # neg-from-pos (abs)
        t5_nfp = t5_p.clone()
        for j,w in enumerate(p3_absneg_w):
            if w != 1.0: t5_nfp[j] = t5_nfp[j] * w

        # negative side
        n3_ids = torch.tensor([n3_tok], dtype=torch.long)
        t5_n   = pipe.text_encoder_3(n3_ids.to(pipe.device))[0].squeeze(0)  # [T5, Ct5]
        t5_n   = t5_n.to(device=pipe.device)
        # base (>=0)
        t5_nb  = t5_n.clone()
        for j,w in enumerate(n3_pos_w):
            if w != 1.0: t5_nb[j] = t5_nb[j] * w
        # pos-from-neg (abs)
        t5_pfn = t5_n.clone()
        for j,w in enumerate(n3_posfrom):
            if w != 1.0: t5_pfn[j] = t5_pfn[j] * w

        # merge routed for T5
        t5_prompt_pre = t5_pb + t5_pfn
        t5_neg_total  = t5_nb + t5_nfp

        # NegPiP dual on T5 token embeddings
        t5_prompt_after, t5_neg_after = _negpip_dual_apply(t5_prompt_pre.unsqueeze(0), t5_neg_total.unsqueeze(0), alpha=1.0)
        # shapes: [1, T5, Ct5]
    else:
        # fallback zeros (keep dtypes/devices aligned)
        t5_prompt_after = torch.zeros(1, 4096, dtype=clip_prompt_after.dtype, device=pipe.device).unsqueeze(0)
        t5_neg_after    = torch.zeros(1, 4096, dtype=clip_neg_after.dtype,    device=pipe.device).unsqueeze(0)

    # --- 7) Merge CLIP (A+B) with T5 as in original SD3 ---
    # pad CLIP last channel to match T5 last channel, then concat along sequence dim (-2)
    # NB: In the original, they pad CLIP (last dim) to T5 dim, then cat([clip, t5], dim=-2).
    clip_prompt_padded = F.pad(clip_prompt_after, (0, t5_prompt_after.shape[-1] - clip_prompt_after.shape[-1]))
    sd3_prompt_embeds  = torch.cat([clip_prompt_padded, t5_prompt_after], dim=-2)

    clip_neg_padded    = F.pad(clip_neg_after, (0, t5_neg_after.shape[-1] - clip_neg_after.shape[-1]))
    sd3_neg_prompt_embeds = torch.cat([clip_neg_padded, t5_neg_after], dim=-2)

    # --- 8) pooled also gets NegPiP dual ---
    pooled_prompt_embeds, negative_pooled_prompt_embeds = _negpip_dual_apply_pooled(
        pooled_prompt_embeds, negative_pooled_prompt_embeds, alpha=1.0
    )

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return sd3_prompt_embeds, sd3_neg_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_flux1(
    pipe: FluxPipeline,
    prompt: str = "",
    prompt2: str = None,
    neg_prompt: str = "",
    neg_prompt2: str = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function can process long prompts with per-token weights (no length limitation)
    for FLUX v1, and extends the behavior with NegPiP routing + dual composition.

    FLUX specifics:
    - Two text encoders:
        * tokenizer / text_encoder      -> CLIP (pooled; FLUX uses pooled text rather than per-token hs)
        * tokenizer_2 / text_encoder_2  -> T5 (token-wise embeddings)
    - CLIP path: pooled representation only (block-averaged pooler_output), **no per-token weighting**.
    - T5 path: per-token linear weighting (no tanh for FLUX).
    - **API change (breaking)**: returns 4 tensors
            (t5_prompt_embeds, clip_prompt_pooled, t5_negative_embeds, clip_negative_pooled)

    NegPiP rules (fixed in this implementation):
    - mode = "dual", negpip_alpha = 1.0, route_negative = True
    - Routing:
        * Positive prompt: tokens with negative weights are removed from the positive stream
            and their absolute weights are sent to the "neg-from-pos" stream.
        * Negative prompt: tokens with negative weights are removed from the negative stream
            and their absolute weights are sent to the "pos-from-neg" stream.
    - Composition (T5 token embeddings):
        * prompt_pre = positive_base + pos_from_neg
        * neg_total  = negative_base + neg_from_pos
        * final t5_prompt_embeds      = prompt_pre - neg_total
        * final t5_negative_embeds    = neg_total + neg_total
        (CLIP pooled side is computed independently for positive / negative via block-average.)

    Tokenization & grouping:
    - CLIP side is grouped into 77-token blocks ([BOS] + 75 tokens + [EOS]) to compute
        per-block pooler_output, then averaged (no weighting).
    - T5 side embeds the whole sequence and applies linear per-token scaling.

    Args:
        pipe (FluxPipeline)
            A diffusers FLUX pipeline. Must provide `tokenizer`, `text_encoder` (CLIP)
            and `tokenizer_2`, `text_encoder_2` (T5).
        prompt (str)
            A prompt string with weights for CLIP/T5. Example: "a (white:1.2) cat, (fur:-0.4)".
            Negative weights here are routed to the negative stream as "neg-from-pos".
        prompt2 (str, optional)
            Alternative prompt string with weights **for T5 only**. If None, falls back to `prompt`.
        neg_prompt (str)
            A negative prompt string with weights. Example: "(blur:1.3), (artifact:-0.7)".
            Negative weights here are routed to the positive stream as "pos-from-neg".
        neg_prompt2 (str, optional)
            Alternative negative prompt string with weights **for T5 only**. If None, falls back to `neg_prompt`.
        device (Optional[str])
            Target device to temporarily move text encoders when the pipeline itself is on CPU.
            If None or "cpu", defaults to "cuda:0" (mirrors the original helper behavior).

    Returns:
        t5_prompt_embeds (torch.Tensor)
            Final positive **T5 token embeddings** after NegPiP dual composition. Shape: [1, T5, Ct5].
        clip_prompt_pooled (torch.Tensor)
            Positive **CLIP pooled embedding** (block-averaged pooler_output). Shape: [1, Cclip].
        t5_negative_embeds (torch.Tensor)
            Final negative **T5 token embeddings** after NegPiP dual composition. Shape: [1, T5, Ct5].
        clip_negative_pooled (torch.Tensor)
            Negative **CLIP pooled embedding** (block-averaged pooler_output). Shape: [1, Cclip].

    Example:
        # NOTE: The return signature has changed (now 4 tensors).
        import torch
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        ).to("cuda:0")

        prompt      = "a (white:1.2) cat in a cozy room, (fur:-0.4)"
        prompt2     = None  # use same text for T5, or provide another string
        neg_prompt  = "(blurry:1.0), (artifact:-0.6)"
        neg_prompt2 = None  # use same text for T5, or provide another string

        t5_pos, clip_pos, t5_neg, clip_neg = get_weighted_text_embeddings_flux1(
            pipe=pipe,
            prompt=prompt,
            prompt2=prompt2,
            neg_prompt=neg_prompt,
            neg_prompt2=neg_prompt2,
            device=None
        )

        # Pass these to your FLUX pipeline call (variable names may differ by pipeline version):
        # images = pipe(
        #     prompt_embeds=t5_pos,
        #     pooled_prompt_embeds=clip_pos,
        #     negative_prompt_embeds=t5_neg,
        #     negative_pooled_prompt_embeds=clip_neg,
        #     generator=torch.Generator(pipe.device).manual_seed(2)
        # ).images
    """
    prompt2 = prompt if prompt2 is None else prompt2
    neg_prompt2 = neg_prompt if neg_prompt2 is None else neg_prompt2

    # choose target device
    target_device = 'cuda:0' if (device is None or device == 'cpu') else device

    # Move encoders to target device only if pipeline itself is on CPU (match original behavior)
    need_move = not pipe.device.type.startswith('cuda')
    if need_move:
        pipe.text_encoder.to(target_device)
        pipe.text_encoder_2.to(target_device)

    # ---------- 1) Tokenize ----------
    # CLIP: tokenizer (prompt / negative)
    p_tok, p_w = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
    n_tok, n_w = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)

    # T5: tokenizer_2 (prompt2 / negative2)
    p2_tok, p2_w = get_prompts_tokens_with_weights_t5(pipe.tokenizer_2, prompt2)
    n2_tok, n2_w = get_prompts_tokens_with_weights_t5(pipe.tokenizer_2, neg_prompt2)

    # ---------- 2) Split weights for routing (both encoders) ----------
    # CLIP weights are not applied, but we still split to keep semantics aligned if needed later
    _, p_pos_w, p_absneg_w = _split_pos_neg_weights(p_w)   # pos base, neg-from-pos
    _, n_pos_w, n_posfrom  = _split_pos_neg_weights(n_w)   # neg base, pos-from-neg

    # T5 weights (used)
    _, p2_pos_w, p2_absneg_w = _split_pos_neg_weights(p2_w)  # pos base, neg-from-pos
    _, n2_pos_w, n2_posfrom  = _split_pos_neg_weights(n2_w)  # neg base, pos-from-neg

    # ---------- 3) CLIP pooled (block-average pooler_output; no per-token weighting) ----------
    # group CLIP tokens into 77-token blocks (same method as original)
    p_groups, _ = group_tokens_and_weights(p_tok.copy(), p_pos_w.copy(), pad_last_block=True)
    n_groups, _ = group_tokens_and_weights(n_tok.copy(), n_pos_w.copy(), pad_last_block=True)

    clip_pooled_list_pos = []
    for token_group in p_groups:
        token_tensor = torch.tensor([token_group], dtype=torch.long, device=target_device)
        with torch.no_grad():
            pe = pipe.text_encoder(token_tensor, output_hidden_states=False)
        pooled = pe.pooler_output.squeeze(0)  # [Cclip]
        clip_pooled_list_pos.append(pooled)
    clip_prompt_pooled = torch.stack(clip_pooled_list_pos, dim=0).mean(dim=0, keepdim=True)  # [1, Cclip]
    clip_prompt_pooled = clip_prompt_pooled.to(dtype=pipe.text_encoder.dtype, device=target_device)

    clip_pooled_list_neg = []
    for token_group in n_groups:
        token_tensor = torch.tensor([token_group], dtype=torch.long, device=target_device)
        with torch.no_grad():
            pe = pipe.text_encoder(token_tensor, output_hidden_states=False)
        pooled = pe.pooler_output.squeeze(0)
        clip_pooled_list_neg.append(pooled)
    clip_negative_pooled = torch.stack(clip_pooled_list_neg, dim=0).mean(dim=0, keepdim=True)  # [1, Cclip]
    clip_negative_pooled = clip_negative_pooled.to(dtype=pipe.text_encoder.dtype, device=target_device)

    # ---------- 4) T5 embeddings (token-wise, linear scaling, with routing + NegPiP dual) ----------
    # Positive side
    p2_ids = torch.tensor([p2_tok], dtype=torch.long, device=target_device)
    with torch.no_grad():
        t5_p = pipe.text_encoder_2(p2_ids)[0].squeeze(0)  # [T5, Ct5]
    t5_p = t5_p.to(device=target_device)

    # base (>=0)
    t5_pos_base = t5_p.clone()
    for j, w in enumerate(p2_pos_w):
        if w != 1.0:
            t5_pos_base[j] = t5_pos_base[j] * w
    # neg-from-pos (abs negatives from positive)
    t5_neg_from_pos = t5_p.clone()
    for j, w in enumerate(p2_absneg_w):
        if w != 1.0:
            t5_neg_from_pos[j] = t5_neg_from_pos[j] * w

    # Negative side
    n2_ids = torch.tensor([n2_tok], dtype=torch.long, device=target_device)
    with torch.no_grad():
        t5_n = pipe.text_encoder_2(n2_ids)[0].squeeze(0)  # [T5, Ct5]
    t5_n = t5_n.to(device=target_device)

    # base (>=0)
    t5_neg_base = t5_n.clone()
    for j, w in enumerate(n2_pos_w):
        if w != 1.0:
            t5_neg_base[j] = t5_neg_base[j] * w
    # pos-from-neg (abs negatives from negative)
    t5_pos_from_neg = t5_n.clone()
    for j, w in enumerate(n2_posfrom):
        if w != 1.0:
            t5_pos_from_neg[j] = t5_pos_from_neg[j] * w

    # Merge routed parts
    t5_prompt_pre = t5_pos_base + t5_pos_from_neg
    t5_neg_total  = t5_neg_base + t5_neg_from_pos

    # NegPiP dual (alpha=1.0) on T5 token embeddings
    t5_prompt_after, t5_negative_after = _negpip_dual_apply(
        t5_prompt_pre.unsqueeze(0),  # [1, T5, Ct5]
        t5_neg_total.unsqueeze(0),   # [1, T5, Ct5]
        alpha=1.0,
    )
    # shapes now: [1, T5, Ct5] each

    # ---------- 5) Restore encoders to CPU if we moved them ----------
    if need_move:
        pipe.text_encoder.to('cpu')
        pipe.text_encoder_2.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    # Return four tensors:
    #   - T5: prompt / negative  (per-token)
    #   - CLIP pooled: prompt / negative (per-sample pooled)
    return t5_prompt_after, clip_prompt_pooled, t5_negative_after, clip_negative_pooled
