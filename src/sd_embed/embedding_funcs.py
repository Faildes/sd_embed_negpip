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
from diffusers import FluxPipeline
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import gc
import logging
from pathlib import Path
import typing
import traceback

try:
    from sd_embed.anima_semantic_prompt import (
        AnimaSemanticPromptFrontend,
        SemanticPromptResult,
        TagLexiconResolver,
    )
except Exception:
    AnimaSemanticPromptFrontend = None
    SemanticPromptResult = None
    TagLexiconResolver = None

logger = logging.getLogger(__name__)

def _negpip_factor(w: torch.Tensor) -> torch.Tensor:
    """
    NegPiP: if w>=0 -> w
            if w<0  -> 1+w (e - |w|*e = (1+w)*e)
    """
    return torch.where(w >= 0, w, 1.0 + w)

def _apply_weights_vec_scale(token_embedding: torch.Tensor, weight_tensor: torch.Tensor) -> torch.Tensor:
    if token_embedding.dim() == 3:
        if token_embedding.size(0) != 1:
            raise ValueError(f"expected (1,L,H) or (L,H), got {tuple(token_embedding.shape)}")
        E = token_embedding[0]
    elif token_embedding.dim() == 2:
        E = token_embedding
    else:
        raise ValueError(f"expected (1,L,H) or (L,H), got {tuple(token_embedding.shape)}")

    if not isinstance(weight_tensor, torch.Tensor):
        w = torch.as_tensor(weight_tensor, device=E.device, dtype=E.dtype)
    else:
        w = weight_tensor.to(device=E.device, dtype=E.dtype)

    if w.numel() != E.size(0):
        raise ValueError(f"weight length mismatch: {w.numel()} vs seq_len {E.size(0)}")

    f = _negpip_factor(w)              # (L,)
    E2 = E * f.unsqueeze(-1)           # (L,H)
    return E2

def _apply_weights_vec_interp_to_last(token_embedding: torch.Tensor, weight_tensor: torch.Tensor) -> torch.Tensor:
    if token_embedding.dim() == 3:
        if token_embedding.size(0) != 1:
            raise ValueError(f"expected (1,L,H) or (L,H), got {tuple(token_embedding.shape)}")
        E = token_embedding[0]
    elif token_embedding.dim() == 2:
        E = token_embedding
    else:
        raise ValueError(f"expected (1,L,H) or (L,H), got {tuple(token_embedding.shape)}")

    if not isinstance(weight_tensor, torch.Tensor):
        w = torch.as_tensor(weight_tensor, device=E.device, dtype=E.dtype)
    else:
        w = weight_tensor.to(device=E.device, dtype=E.dtype)

    if w.numel() != E.size(0):
        raise ValueError(f"weight length mismatch: {w.numel()} vs seq_len {E.size(0)}")

    f = _negpip_factor(w)                              # (L,)
    anchor = E[-1].unsqueeze(0).expand_as(E)          # (L,H)
    E2 = anchor + (E - anchor) * f.unsqueeze(-1)      # (L,H)
    return E2


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
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion v1.5
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    original_clip_layers = pipe.text_encoder.text_model.encoder.layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]
    
    eos = pipe.tokenizer.eos_token_id 
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups ,prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working
    # we must embed prompt group by group
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        token_embedding = pipe.text_encoder(token_tensor)[0].squeeze(0) 
        token_embedding = _apply_weights_vec_scale(token_embedding.squeeze(0), weight_tensor).unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        neg_token_embedding = pipe.text_encoder(neg_token_tensor)[0].squeeze(0) 
        neg_token_embedding = _apply_weights_vec_scale(neg_token_embedding.squeeze(0), neg_weight_tensor).unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds       = torch.cat(embeds, dim = 1)
    neg_prompt_embeds   = torch.cat(neg_embeds, dim = 1)
    
    # recover clip layers
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
    pipe: StableDiffusionXLPipeline
    , prompt : str                  = ""
    , neg_prompt: str               = ""
    , pad_last_block                = True
    , lora_scale: Optional[float]   = None
    , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    import math
    eos = pipe.tokenizer.eos_token_id 
    dynamically_scale_lora_layers(pipe, lora_scale = lora_scale)
    
    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = pipe.device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = get_prompt_hidden_states_sdxl(prompt_embeds_1, clip_skip=clip_skip)

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(prompt_embeds_2, clip_skip=clip_skip)
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        token_embedding = _apply_weights_vec_scale(token_embedding.squeeze(0), weight_tensor).unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = get_prompt_hidden_states_sdxl(neg_prompt_embeds_1, clip_skip=clip_skip)        

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(neg_prompt_embeds_2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        neg_token_embedding = _apply_weights_vec_scale(neg_token_embedding.squeeze(0), neg_weight_tensor).unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    dynamically_unscale_lora_layers(pipe, lora_scale = lora_scale)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def get_weighted_text_embeddings_sdxl_refiner(
    pipe: StableDiffusionXLPipeline
    , prompt : str                  = ""
    , neg_prompt: str               = ""
    , lora_scale: Optional[float]   = None
    , clip_skip: Optional[int]      = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    import math
    eos = 49407 #pipe.tokenizer.eos_token_id 
    dynamically_scale_lora_layers(pipe, lora_scale = lora_scale)
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups_2)):
        # get positive prompt embeddings with weights        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = pipe.device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(prompt_embeds_2, clip_skip=clip_skip)
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)
        token_embedding = _apply_weights_vec_interp_to_last(token_embedding, weight_tensor_2).unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(neg_prompt_embeds_2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)
        neg_token_embedding = _apply_weights_vec_interp_to_last(neg_token_embedding, neg_weight_tensor_2).unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    dynamically_unscale_lora_layers(pipe, lora_scale = lora_scale)
    
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
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL, support two prompt sets.
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    prompt_2        = prompt_2 or prompt
    neg_prompt_2    = neg_prompt_2 or neg_prompt
    
    import math
    eos = pipe.tokenizer.eos_token_id
    
    dynamically_scale_lora_layers(pipe, lora_scale = lora_scale)
    
    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt_2
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt_2
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    # now, need to ensure prompt and prompt_2 has the same lemgth
    prompt_token_len        = len(prompt_tokens)
    prompt_token_len_2      = len(prompt_tokens_2)
    if prompt_token_len > prompt_token_len_2:
        prompt_tokens_2     = prompt_tokens_2   + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights_2    = prompt_weights_2  + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    else:
        prompt_tokens       = prompt_tokens     + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights      = prompt_weights    + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    
    # now, need to ensure neg_prompt and net_prompt_2 has the same lemgth
    neg_prompt_token_len        = len(neg_prompt_tokens)
    neg_prompt_token_len_2      = len(neg_prompt_tokens_2)
    if neg_prompt_token_len > neg_prompt_token_len_2:
        neg_prompt_tokens_2     = neg_prompt_tokens_2   + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights_2    = neg_prompt_weights_2  + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    else:
        neg_prompt_tokens       = neg_prompt_tokens     + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights      = neg_prompt_weights    + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , device    = pipe.device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            , device = pipe.device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , device    = pipe.device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = get_prompt_hidden_states_sdxl(prompt_embeds_1, clip_skip=clip_skip)

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(prompt_embeds_2, clip_skip=clip_skip)
        pooled_prompt_embeds = prompt_embeds_2[0]
        
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.squeeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.squeeze(0)
        prompt_embeds_1_hidden_states = _apply_weights_vec_interp_to_last(prompt_embeds_1_hidden_states, weight_tensor)
        prompt_embeds_2_hidden_states = _apply_weights_vec_interp_to_last(prompt_embeds_2_hidden_states, weight_tensor_2)
                              
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.unsqueeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.unsqueeze(0)
        
        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.cat(prompt_embeds_list, dim=-1)
        
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , device = pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , device    = pipe.device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , device    = pipe.device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = get_prompt_hidden_states_sdxl(neg_prompt_embeds_1, clip_skip=clip_skip)

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = get_prompt_hidden_states_sdxl(neg_prompt_embeds_2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.squeeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.squeeze(0)
        neg_prompt_embeds_1_hidden_states = _apply_weights_vec_interp_to_last(neg_prompt_embeds_1_hidden_states, neg_weight_tensor)
        neg_prompt_embeds_2_hidden_states = _apply_weights_vec_interp_to_last(neg_prompt_embeds_2_hidden_states, neg_weight_tensor_2)
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.unsqueeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.unsqueeze(0)

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.cat(neg_prompt_embeds_list, dim=-1)
        
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    dynamically_unscale_lora_layers(pipe, lora_scale = lora_scale)
    
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
     This function can process long prompt with weights, no length limitation
     for Stable Cascade

     Args:
         pipe (typing.Union[StableCascadePriorPipeline, StableCascadeDecoderPipeline])
         prompt (str)
         neg_prompt (str)
     Returns:
         prompt_embeds (torch.Tensor)
         neg_prompt_embeds (torch.Tensor)
         pooled_prompt_embeds (torch.Tensor)
         negative_pooled_prompt_embeds (torch.Tensor)
     """
    import math
    eos = pipe.tokenizer.eos_token_id
    
    dynamically_scale_lora_layers(pipe, lora_scale = lora_scale)

    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )

    # padding the shorter one
    prompt_token_len = len(prompt_tokens)
    neg_prompt_token_len = len(neg_prompt_tokens)

    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens = (
                neg_prompt_tokens +
                [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights = (
                neg_prompt_weights +
                [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens = (
                prompt_tokens
                + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights = (
                prompt_weights
                + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )

    embeds = []
    neg_embeds = []

    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block=pad_last_block
    )

    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block=pad_last_block
    )

    # get prompt embeddings one by one is not working.
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            , dtype=torch.long, device=pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype=torch.float16
            , device=pipe.device
        )

        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        prompt_embeds_1_hidden_states = get_prompt_hidden_states_s_cascade(prompt_embeds_1, clip_skip=clip_skip).cpu()

        pooled_prompt_embeds = prompt_embeds_1.text_embeds.unsqueeze(1)

        prompt_embeds_list = [prompt_embeds_1_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        token_embedding = _apply_weights_vec_scale(token_embedding.squeeze(0), weight_tensor).unsqueeze(0)

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding.cpu())

        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype=torch.long, device=pipe.device
        )

        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype=torch.float16
            , device=pipe.device
        )

        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = get_prompt_hidden_states_s_cascade(neg_prompt_embeds_1, clip_skip=clip_skip).cpu()
        negative_pooled_prompt_embeds = neg_prompt_embeds_1.text_embeds.unsqueeze(1)

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        neg_token_embedding = _apply_weights_vec_scale(neg_token_embedding.squeeze(0), neg_weight_tensor).unsqueeze(0)

        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding.cpu())

    prompt_embeds = torch.cat(embeds, dim=1).to(pipe.device)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1).to(pipe.device)

    dynamically_unscale_lora_layers(pipe, lora_scale = lora_scale)

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
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion 3
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        sd3_prompt_embeds (torch.Tensor)
        sd3_neg_prompt_embeds (torch.Tensor)
        pooled_prompt_embeds (torch.Tensor)
        negative_pooled_prompt_embeds (torch.Tensor)
    """
    import math
    eos = pipe.tokenizer.eos_token_id 
    
    dynamically_scale_lora_layers(pipe, lora_scale = lora_scale)
    
    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # tokenizer 3
    prompt_tokens_3, prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, prompt
    )

    neg_prompt_tokens_3, neg_prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = pipe.device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = get_prompt_hidden_states_sd3(prompt_embeds_1, clip_skip=clip_skip)
        pooled_prompt_embeds_1 = prompt_embeds_1[0]

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = get_prompt_hidden_states_sd3(prompt_embeds_2, clip_skip=clip_skip)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        token_embedding = _apply_weights_vec_scale(token_embedding.squeeze(0), weight_tensor).unsqueeze(0)

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = get_prompt_hidden_states_sd3(neg_prompt_embeds_1, clip_skip=clip_skip)
        negative_pooled_prompt_embeds_1 = neg_prompt_embeds_1[0]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = get_prompt_hidden_states_sd3(neg_prompt_embeds_2, clip_skip=clip_skip)
        negative_pooled_prompt_embeds_2 = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        neg_token_embedding = _apply_weights_vec_scale(neg_token_embedding.squeeze(0), neg_weight_tensor).unsqueeze(0)
                
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1)
    
    if use_t5_encoder and pipe.text_encoder_3:        
        # ----------------- generate positive t5 embeddings --------------------
        prompt_tokens_3 = torch.tensor([prompt_tokens_3],dtype=torch.long)
        
        t5_prompt_embeds    = pipe.text_encoder_3(prompt_tokens_3.to(pipe.device))[0].squeeze(0)
        t5_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)
        
        # add weight to t5 prompt
        f3 = _negpip_factor(torch.tensor(prompt_weights_3, dtype=t5_prompt_embeds.dtype, device=t5_prompt_embeds.device))
        t5_prompt_embeds = t5_prompt_embeds * f3.unsqueeze(-1)
        t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0)
    else:
        t5_prompt_embeds    = torch.zeros(1, 4096, dtype = prompt_embeds.dtype).unsqueeze(0)
        t5_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)
        
    # merge with the clip embedding 1 and clip embedding 2
    clip_prompt_embeds = torch.nn.functional.pad(
        prompt_embeds, (0, t5_prompt_embeds.shape[-1] - prompt_embeds.shape[-1])
    )
    sd3_prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    
    if use_t5_encoder and pipe.text_encoder_3:  
        # ---------------------- get neg t5 embeddings -------------------------
        neg_prompt_tokens_3 = torch.tensor([neg_prompt_tokens_3],dtype=torch.long)
        
        t5_neg_prompt_embeds    = pipe.text_encoder_3(neg_prompt_tokens_3.to(pipe.device))[0].squeeze(0)
        t5_neg_prompt_embeds    = t5_neg_prompt_embeds.to(device=pipe.device)
        
        # add weight to neg t5 embeddings
        fn3 = _negpip_factor(torch.tensor(neg_prompt_weights_3, dtype=t5_neg_prompt_embeds.dtype, device=t5_neg_prompt_embeds.device))
        t5_neg_prompt_embeds = t5_neg_prompt_embeds * fn3.unsqueeze(-1)
        t5_neg_prompt_embeds = t5_neg_prompt_embeds.unsqueeze(0)
    else: 
        t5_neg_prompt_embeds    = torch.zeros(1, 4096, dtype = prompt_embeds.dtype).unsqueeze(0)
        t5_neg_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)

    clip_neg_prompt_embeds = torch.nn.functional.pad(
        negative_prompt_embeds, (0, t5_neg_prompt_embeds.shape[-1] - negative_prompt_embeds.shape[-1])
    )
    sd3_neg_prompt_embeds = torch.cat([clip_neg_prompt_embeds, t5_neg_prompt_embeds], dim=-2)
    
    # padding 
    import torch.nn.functional as F
    size_diff = sd3_neg_prompt_embeds.size(1) - sd3_prompt_embeds.size(1)
    # Calculate padding. Format for pad is (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    # Since we are padding along the second dimension (axis=1), we need (0, 0, padding_top, padding_bottom, 0, 0)
    # Here padding_top will be 0 and padding_bottom will be size_diff

    # Check if padding is needed
    if size_diff > 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_prompt_embeds = F.pad(sd3_prompt_embeds, padding)
    elif size_diff < 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_neg_prompt_embeds = F.pad(sd3_neg_prompt_embeds, padding)
    
    dynamically_unscale_lora_layers(pipe, lora_scale = lora_scale)
    
    return sd3_prompt_embeds, sd3_neg_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_flux1(
    pipe: FluxPipeline
    , prompt: str       = ""
    , prompt2: str      = None
    , device            = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function can process long prompt with weights for flux1 model
    
    Args:
        pipe (['FluxPipeline']): The FluxPipeline
        prompt (['string']): the 1st prompt
        prompt2 (['string']): the 2nd prompt
        device (['string']): target device
    Returns:
        A tuple include two embedding tensors
    """
    prompt2 = prompt if prompt2 is None else prompt2
    
    # so that user can assign custom cuda
    if device is None or device == 'cpu':
        target_device = 'cuda:0'
    else:
        target_device = device
        
    # prepare text_encoder and text_encoder_2
    if not pipe.device.type.startswith('cuda'):
        pipe.text_encoder.to(target_device)
        pipe.text_encoder_2.to(target_device)
    
    # tokenizer 1 - openai/clip-vit-large-patch14
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    
    # tokenizer 2 - google/t5-v1_1-xxl
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_2, prompt2
    )
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = True
    )
        
    # # get positive prompt embeddings, flux1 use only text_encoder 1 pooled embeddings
    # token_tensor = torch.tensor(
    #     [prompt_token_groups[0]]
    #     , dtype = torch.long, device = device
    # )
    # # use first text encoder
    # prompt_embeds_1 = pipe.text_encoder(
    #     token_tensor.to(device)
    #     , output_hidden_states  = False
    # )
    # pooled_prompt_embeds_1  = prompt_embeds_1.pooler_output
    # prompt_embeds           = pooled_prompt_embeds_1.to(dtype = pipe.text_encoder.dtype, device = device)
    
    # use avg pooling embeddings
    pool_embeds_list = []
    for token_group in prompt_token_groups:
        token_tensor = torch.tensor(
            [token_group]
            , dtype = torch.long
            , device = target_device
        )
        with torch.no_grad():
            prompt_embeds_1 = pipe.text_encoder(
                token_tensor.to(target_device)
                , output_hidden_states  = False
            )
        pooled_prompt_embeds = prompt_embeds_1.pooler_output.squeeze(0)
        pool_embeds_list.append(pooled_prompt_embeds)
        
    prompt_embeds = torch.stack(pool_embeds_list,dim=0)
    
    # get the avg pool
    prompt_embeds = prompt_embeds.mean(dim=0, keepdim=True)
    # prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(dtype = pipe.text_encoder.dtype, device = target_device)
            
    # generate positive t5 embeddings 
    prompt_tokens_2 = torch.tensor([prompt_tokens_2],dtype=torch.long)
    
    with torch.no_grad():
        t5_prompt_embeds    = pipe.text_encoder_2(prompt_tokens_2.to(target_device))[0].squeeze(0)
    t5_prompt_embeds    = t5_prompt_embeds.to(device = target_device)
    
    # add weight to t5 prompt
    f2 = _negpip_factor(torch.tensor(prompt_weights_2, dtype=t5_prompt_embeds.dtype, device=t5_prompt_embeds.device))
    t5_prompt_embeds = t5_prompt_embeds * f2.unsqueeze(-1)
    t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0)
    t5_prompt_embeds = t5_prompt_embeds.to(dtype = pipe.text_encoder_2.dtype, device = target_device)
    
    # release text encoder from vram
    if pipe.device.type.startswith('cpu'):
        pipe.text_encoder.to('cpu')
        pipe.text_encoder_2.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
    
    return t5_prompt_embeds,prompt_embeds

# ---------------------------
# helpers
# ---------------------------

def _find_sublist(haystack: List[int], needle: List[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    end = len(haystack) - n + 1
    for i in range(start, end):
        if haystack[i:i+n] == needle:
            return i
    return -1

def _strip_attention_syntax(prompt: str) -> Tuple[str, List[Tuple[str, float]]]:
    segs = parse_prompt_attention(prompt or "")
    clean = "".join(t for t, _ in segs)
    return clean, [(t, float(w)) for t, w in segs]

def _decode_piece(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([tid])

def _zimage_token_weights_from_segments(tokenizer, prompt: str) -> Tuple[List[int], List[float], str]:
    clean_text, segs = _strip_attention_syntax(prompt)
    if not clean_text:
        return [], [], clean_text

    enc = tokenizer(clean_text, add_special_tokens=False, truncation=False)
    ids = enc.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]

    pieces = [_decode_piece(tokenizer, tid) for tid in ids]
    recon = "".join(pieces)

    if recon != clean_text:
        token_ids: List[int] = []
        weights: List[float] = []
        for t, w in segs:
            if not t:
                continue
            e = tokenizer(t, add_special_tokens=False, truncation=False)
            tid = e.input_ids
            if isinstance(tid, torch.Tensor):
                tid = tid.tolist()
            if isinstance(tid, list) and len(tid) == 1 and isinstance(tid[0], list):
                tid = tid[0]
            token_ids.extend(tid)
            weights.extend([float(w)] * len(tid))
        return token_ids, weights, clean_text

    spans: List[Tuple[int, int, float]] = []
    pos = 0
    for t, w in segs:
        if t is None:
            continue
        ln = len(t)
        if ln > 0:
            spans.append((pos, pos + ln, float(w)))
        pos += ln

    weights: List[float] = []
    tpos = 0
    span_i = 0
    for piece in pieces:
        ts, te = tpos, tpos + len(piece)
        tpos = te

        if te <= ts:
            weights.append(1.0)
            continue

        while span_i < len(spans) and spans[span_i][1] <= ts:
            span_i += 1

        wsum = 0.0
        osum = 0
        j = span_i
        while j < len(spans) and spans[j][0] < te:
            s0, s1, w = spans[j]
            ov = max(0, min(te, s1) - max(ts, s0))
            if ov > 0:
                wsum += w * ov
                osum += ov
            j += 1

        weights.append(wsum / osum if osum > 0 else 1.0)

    return ids, weights, clean_text

def _apply_weights_interp_mean(
    hidden: torch.Tensor,
    weight_tensor: torch.Tensor,
    content_slice: Tuple[int, int],
    strength: float = 1.6,
    clamp_min: float = 0.0,
    clamp_max: float = 3.0,
) -> torch.Tensor:
    # NegPiP factor
    f = _negpip_factor(weight_tensor)  # (L,)

    f = 1.0 + (f - 1.0) * strength
    if clamp_min is not None and clamp_max is not None:
        f = f.clamp(min=clamp_min, max=clamp_max)

    c0, c1 = content_slice
    if 0 <= c0 < c1 <= hidden.size(0):
        anchor = hidden[c0:c1].mean(dim=0, keepdim=True)  # (1,H)
    else:
        anchor = hidden.mean(dim=0, keepdim=True)

    return anchor + (hidden - anchor) * f.unsqueeze(-1)

# ---------------------------
# main (drop-in replacement)
# ---------------------------

def get_weighted_text_embeddings_zimage(
    pipe,
    prompt: Union[str, List[str]]        = "",
    neg_prompt: Union[str, List[str]]    = "",
    max_sequence_length: int             = 1024,
    lora_scale: Optional[float]          = None,
    enable_thinking: bool                = False,
    weight_strength: float               = 1024.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if isinstance(prompt, str):
        prompt_list: List[str] = [prompt]
    else:
        prompt_list = list(prompt)

    if isinstance(neg_prompt, str):
        neg_list: List[str] = [neg_prompt] * len(prompt_list)
    else:
        neg_list = list(neg_prompt)
        if len(neg_list) != len(prompt_list):
            raise ValueError(f"The number of prompts and neg_prompts not matched: {len(prompt_list)} / {len(neg_list)}")

    device = pipe.device
    tokenizer = pipe.tokenizer

    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    MARK_S = "<<<PROMPT_START>>>"
    MARK_E = "<<<PROMPT_END>>>"

    ms_ids = tokenizer(MARK_S, add_special_tokens=False, truncation=False).input_ids
    me_ids = tokenizer(MARK_E, add_special_tokens=False, truncation=False).input_ids
    if isinstance(ms_ids, torch.Tensor):
        ms_ids = ms_ids.tolist()
    if isinstance(me_ids, torch.Tensor):
        me_ids = me_ids.tolist()
    if isinstance(ms_ids, list) and len(ms_ids) == 1 and isinstance(ms_ids[0], list):
        ms_ids = ms_ids[0]
    if isinstance(me_ids, list) and len(me_ids) == 1 and isinstance(me_ids[0], list):
        me_ids = me_ids[0]
        
    def _has_top_level_AND(text: str) -> bool:
        if "AND" not in (text or ""):
            return False
        pr = br = 0
        i = 0
        s = text
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "(":
                pr += 1
            elif ch == ")" and pr > 0:
                pr -= 1
            elif ch == "[":
                br += 1
            elif ch == "]" and br > 0:
                br -= 1

            if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
                prev = s[i-1] if i > 0 else " "
                nxt  = s[i+3] if i + 3 < n else " "
                prev_ok = prev.isspace() or (not prev.isalnum() and prev != "_")
                nxt_ok  = nxt.isspace()  or (not nxt.isalnum()  and nxt != "_")
                if prev_ok and nxt_ok:
                    return True
            i += 1
        return False


    def _split_top_level_AND(text: str) -> List[str]:
        out, buf = [], []
        pr = br = 0
        i = 0
        s = text or ""
        n = len(s)

        def boundary(ch: str) -> bool:
            return ch.isspace() or (not ch.isalnum() and ch != "_")

        while i < n:
            ch = s[i]
            if ch == "\\" and i + 1 < n:
                buf.append(s[i]); buf.append(s[i+1])
                i += 2
                continue

            if ch == "(":
                pr += 1
            elif ch == ")" and pr > 0:
                pr -= 1
            elif ch == "[":
                br += 1
            elif ch == "]" and br > 0:
                br -= 1

            if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
                prev = s[i-1] if i > 0 else " "
                nxt  = s[i+3] if i + 3 < n else " "
                if boundary(prev) and boundary(nxt):
                    seg = "".join(buf).strip()
                    if seg:
                        out.append(seg)
                    buf = []
                    i += 3
                    continue

            buf.append(ch)
            i += 1

        seg = "".join(buf).strip()
        if seg or not out:
            out.append(seg)
        return out


    def _split_suffix_weight_top_level(seg: str) -> Tuple[str, float]:
        s = (seg or "").strip()
        if not s:
            return "", 1.0
        pr = br = 0
        last_colon = -1
        for i, ch in enumerate(s):
            if ch == "\\":
                continue
            if ch == "(":
                pr += 1
            elif ch == ")" and pr > 0:
                pr -= 1
            elif ch == "[":
                br += 1
            elif ch == "]" and br > 0:
                br -= 1
            elif ch == ":" and pr == 0 and br == 0:
                last_colon = i
        if last_colon == -1:
            return s, 1.0

        left = s[:last_colon].strip()
        right = s[last_colon+1:].strip()
        try:
            w = float(right)
        except Exception:
            return s, 1.0
        if not left:
            return s, 1.0
        return left, w


    def _encode_single_meta(text: str):
        content_ids, content_w, clean_text = _zimage_token_weights_from_segments(tokenizer, text)
        user_text = f"{MARK_S}{clean_text}{MARK_E}"

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_text}]
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            templated = user_text

        text_inputs = tokenizer(
            templated,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)
        attn_mask = text_inputs.attention_mask.to(device).bool()

        with torch.no_grad():
            outputs = pipe.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        hidden_all = outputs.hidden_states[-2][0]     # (T,H)
        valid_mask = attn_mask[0]
        full_ids = input_ids[0][valid_mask]
        hidden = hidden_all[valid_mask]

        full_ids_list = full_ids.tolist()
        s_idx = _find_sublist(full_ids_list, ms_ids, start=0)
        if s_idx != -1:
            s_idx += len(ms_ids)
        e_idx = _find_sublist(full_ids_list, me_ids, start=max(0, s_idx))
        if e_idx == -1:
            e_idx = s_idx + (len(content_w) if content_w else 0)

        w_full = [1.0] * len(full_ids_list)
        if s_idx != -1 and content_w:
            n = max(0, min(len(content_w), len(w_full) - s_idx, max(0, e_idx - s_idx)))
            for i in range(n):
                w_full[s_idx + i] = float(content_w[i])

        weight_tensor = torch.tensor(w_full, dtype=hidden.dtype, device=hidden.device)

        hidden = _apply_weights_interp_mean(
            hidden,
            weight_tensor,
            content_slice=(s_idx if s_idx != -1 else 0, min(e_idx, hidden.size(0))),
            strength=weight_strength,
        )

        s = max(0, min(int(s_idx if s_idx != -1 else 0), hidden.size(0)))
        e = max(s, min(int(e_idx), hidden.size(0)))
        return hidden, s, e


    def _resample_seq_to_len(x: torch.Tensor, L: int) -> torch.Tensor:
        if x.size(0) == L:
            return x
        if x.size(0) <= 1:
            return x.repeat(L, 1)
        import torch.nn.functional as F
        t = x.transpose(0, 1).unsqueeze(0)          # (1,H,S)
        t = F.interpolate(t, size=L, mode="linear", align_corners=False)
        return t.squeeze(0).transpose(0, 1)         # (L,H)


    def _encode_with_AND_safe(text: str, and_strength: float = 0.6, base_bias: float = 4.0) -> torch.Tensor:
        if not _has_top_level_AND(text):
            h, _, _ = _encode_single_meta(text)
            return h

        base_h, base_s, base_e = _encode_single_meta(text)
        if and_strength <= 0.0 or (base_e - base_s) <= 0:
            return base_h

        parts = _split_top_level_AND(text)
        parsed = []
        for p in parts:
            t, w = _split_suffix_weight_top_level(p)
            t = (t or "").strip()
            if t:
                parsed.append((t, float(w)))
        if len(parsed) <= 1:
            return base_h

        base_len = base_e - base_s
        base_content = base_h[base_s:base_e]

        contents = [base_content]
        weights  = [float(base_bias)]

        for t, w in parsed:
            hi, si, ei = _encode_single_meta(t)
            ci = hi[si:ei]
            if ci.numel() == 0:
                continue
            ci = _resample_seq_to_len(ci, base_len)
            contents.append(ci)
            weights.append(float(w))

        denom = sum(abs(w) for w in weights) or 1.0
        mixed = torch.zeros_like(base_content)
        for c, w in zip(contents, weights):
            mixed = mixed + c * float(w)
        mixed = mixed / float(denom)

        mod = base_h.clone()
        mod[base_s:base_e] = mixed

        return base_h + (mod - base_h) * float(and_strength)

    prompt_embeds: List[torch.Tensor] = [_encode_with_AND_safe(p, and_strength=0.6, base_bias=4.0) for p in prompt_list]
    neg_embeds: List[torch.Tensor]    = [_encode_with_AND_safe(n, and_strength=0.6, base_bias=4.0) for n in neg_list]

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
    return prompt_embeds, neg_embeds

# ============================================================
# Anima (AM) embeddings
# ============================================================

# Notes:
# - Public Anima Preview pipelines expose tokenizer/t5_tokenizer/llm_adapter.
# - Some custom AnimaPipeline builds expose prompt_tokenizer and hide/merge the adapter
#   into text_encoder/encode_prompt.  The helpers below support both forms.


def _anima_get_component(pipe, names: List[str], *, required: bool = False):
    comps = getattr(pipe, "components", None)
    for name in names:
        try:
            value = getattr(pipe, name)
            if value is not None:
                return value
        except Exception:
            pass
        try:
            if isinstance(comps, dict) and name in comps and comps[name] is not None:
                return comps[name]
        except Exception:
            pass
    if required:
        raise AttributeError(
            f"AnimaPipeline is missing required component. Tried: {', '.join(names)}"
        )
    return None


def _anima_get_device(pipe):
    for name in ("_execution_device", "execution_device", "_anima_execution_device", "device"):
        try:
            v = getattr(pipe, name)
            if v is not None:
                return v
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _anima_get_dtype(pipe):
    for obj_name in ("text_encoder", "transformer"):
        obj = _anima_get_component(pipe, [obj_name], required=False)
        try:
            v = getattr(obj, "dtype")
            if v is not None:
                return v
        except Exception:
            pass
    for name in ("text_encoder_dtype", "_anima_model_dtype", "model_dtype", "dtype"):
        try:
            v = getattr(pipe, name)
            if v is not None:
                return v
        except Exception:
            pass
    return torch.float16


def _anima_call_encode_prompt(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
) -> torch.Tensor:
    """Call pipe.encode_prompt while tolerating custom signatures."""
    import inspect

    if not hasattr(pipe, "encode_prompt"):
        raise AttributeError("AnimaPipeline has no encode_prompt method")

    fn = pipe.encode_prompt
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())

    kwargs = {}
    if "prompt" in allowed:
        kwargs["prompt"] = text
    else:
        # Very unusual, but keep a helpful failure instead of silently returning wrong embeddings.
        raise TypeError("pipe.encode_prompt does not accept a 'prompt' argument")

    if "negative_prompt" in allowed:
        kwargs["negative_prompt"] = ""
    if "do_classifier_free_guidance" in allowed:
        kwargs["do_classifier_free_guidance"] = True
    if "num_images_per_prompt" in allowed:
        kwargs["num_images_per_prompt"] = int(num_images_per_prompt)
    if "max_sequence_length" in allowed:
        kwargs["max_sequence_length"] = int(max_sequence_length)
    if "device" in allowed:
        kwargs["device"] = _anima_get_device(pipe)
    if "dtype" in allowed:
        kwargs["dtype"] = _anima_get_dtype(pipe)

    out = fn(**kwargs)
    if isinstance(out, (tuple, list)):
        if not out:
            raise RuntimeError("pipe.encode_prompt returned an empty tuple/list")
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"pipe.encode_prompt returned {type(out)!r}, expected torch.Tensor or tuple/list")
    return out


def _anima_encode_cond_only(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
) -> torch.Tensor:
    return _anima_call_encode_prompt(
        pipe,
        text,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )


def _has_top_level_AND(text: str) -> bool:
    if "AND" not in (text or ""):
        return False
    pr = br = 0
    i = 0
    s = text
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1

        if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
            prev = s[i-1] if i > 0 else " "
            nxt  = s[i+3] if i + 3 < n else " "
            prev_ok = prev.isspace() or (not prev.isalnum() and prev != "_")
            nxt_ok  = nxt.isspace()  or (not nxt.isalnum()  and nxt != "_")
            if prev_ok and nxt_ok:
                return True
        i += 1
    return False


def _split_top_level_AND(text: str) -> List[str]:
    out, buf = [], []
    pr = br = 0
    i = 0
    s = text or ""
    n = len(s)

    def boundary(ch: str) -> bool:
        return ch.isspace() or (not ch.isalnum() and ch != "_")

    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i]); buf.append(s[i+1])
            i += 2
            continue

        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1

        if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
            prev = s[i-1] if i > 0 else " "
            nxt  = s[i+3] if i + 3 < n else " "
            if boundary(prev) and boundary(nxt):
                seg = "".join(buf).strip()
                if seg:
                    out.append(seg)
                buf = []
                i += 3
                continue

        buf.append(ch)
        i += 1

    seg = "".join(buf).strip()
    if seg or not out:
        out.append(seg)
    return out


def _split_suffix_weight_top_level(seg: str) -> Tuple[str, float]:
    s = (seg or "").strip()
    if not s:
        return "", 1.0
    pr = br = 0
    last_colon = -1
    for i, ch in enumerate(s):
        if ch == "\\":
            continue
        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1
        elif ch == ":" and pr == 0 and br == 0:
            last_colon = i

    if last_colon == -1:
        return s, 1.0

    left = s[:last_colon].strip()
    right = s[last_colon+1:].strip()
    try:
        w = float(right)
    except Exception:
        return s, 1.0
    if not left:
        return s, 1.0
    return left, w


def _pick_weighted_segments_for_mix(
    segs: List[Tuple[str, float]],
    *,
    max_segments: int = 12,
    min_len: int = 2,
) -> List[Tuple[str, float]]:
    cand = []
    for t, w in segs:
        tt = (t or "").strip()
        if not tt or len(tt) < min_len:
            continue
        ww = float(w)
        if abs(ww - 1.0) < 1e-6:
            continue
        score = abs(ww - 1.0) * max(1, len(tt))
        cand.append((score, tt, ww))

    if not cand:
        return []

    cand.sort(key=lambda x: x[0], reverse=True)
    cand = cand[:max_segments]
    return [(t, w) for _, t, w in cand]


def _anima_v3_resolve_text_encoder_backbone(text_encoder):
    """Return the hidden-state backbone for supported Anima Qwen encoders.

    Original Anima commonly exposes Qwen3-0.6B-Base as a bare Qwen3Model, while
    Qwen3.5-0.8B-Base may be exposed through a causal-LM wrapper whose `.model`
    is the actual text backbone. This is intentionally independent from the
    28/40-block image-transformer selection.
    """
    direct = getattr(text_encoder, "language_model", None)
    if direct is not None:
        return direct
    model = getattr(text_encoder, "model", None)
    nested = getattr(model, "language_model", None) if model is not None else None
    if nested is not None:
        return nested
    if model is not None and hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    return text_encoder


@torch.no_grad()
def _anima_encode_direct_if_possible(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: float = 0.0,
    weight_clamp_max: float = 3.0,
) -> Optional[torch.Tensor]:
    """Use the public tokenizer/t5_tokenizer/llm_adapter path when available.

    Returns None for custom pipelines where the adapter is hidden/merged; callers then
    use encode_prompt + final-conditioning weighting.
    """
    tokenizer = _anima_get_component(pipe, ["tokenizer", "prompt_tokenizer"], required=False)
    t5_tokenizer = _anima_get_component(pipe, ["t5_tokenizer", "target_tokenizer"], required=False)
    text_encoder = _anima_get_component(pipe, ["text_encoder"], required=False)
    llm_adapter = _anima_get_component(pipe, ["llm_adapter", "adapter", "text_adapter"], required=False)

    if tokenizer is None or t5_tokenizer is None or text_encoder is None or llm_adapter is None:
        return None

    device = _anima_get_device(pipe)
    dtype = _anima_get_dtype(pipe)

    text = text or ""
    clean_text, _ = _strip_attention_syntax(text)
    if clean_text.strip() == "":
        target_dim = int(getattr(getattr(llm_adapter, "config", None), "target_dim", 1024))
        return torch.zeros(
            int(num_images_per_prompt),
            512,
            target_dim,
            device=device,
            dtype=dtype,
        )

    qwen_ids, qwen_weights, qwen_mask, _ = _token_weights_from_offsets(
        tokenizer,
        text,
        max_sequence_length=max_sequence_length,
    )
    t5_ids, t5_weights, _t5_mask, _ = _token_weights_from_offsets(
        t5_tokenizer,
        text,
        max_sequence_length=max_sequence_length,
    )

    qwen_input_ids = torch.tensor([qwen_ids], dtype=torch.long, device=device)
    qwen_attention_mask = torch.tensor([qwen_mask], dtype=torch.long, device=device)
    qwen_weight_tensor = torch.tensor([qwen_weights], dtype=dtype, device=device)
    t5_input_ids = torch.tensor([t5_ids], dtype=torch.long, device=device)
    t5_weight_tensor = torch.tensor([t5_weights], dtype=dtype, device=device)

    cache = getattr(pipe, "_sd_embed_anima_empty_cache", None)
    cache_key = ("direct", id(tokenizer), id(t5_tokenizer), id(llm_adapter), str(device), str(dtype), int(max_sequence_length))
    if cache is None:
        cache = {}
        setattr(pipe, "_sd_embed_anima_empty_cache", cache)

    if cache_key not in cache:
        empty_qwen = tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
            return_tensors="pt",
        )
        empty_t5 = t5_tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
            return_tensors="pt",
        )
        empty_qwen_ids = empty_qwen.input_ids.to(device)
        empty_qwen_mask = empty_qwen.attention_mask.to(device)
        empty_t5_ids = empty_t5.input_ids.to(device)

        text_backbone = _anima_v3_resolve_text_encoder_backbone(text_encoder)
        empty_qwen_out = text_backbone(
            input_ids=empty_qwen_ids,
            attention_mask=empty_qwen_mask,
        )
        empty_qwen_hidden = empty_qwen_out.last_hidden_state.to(dtype=dtype)
        empty_adapted = llm_adapter(
            source_hidden_states=empty_qwen_hidden,
            target_input_ids=empty_t5_ids,
        ).to(dtype=dtype)
        empty_adapted = _pad_or_crop_anima_embeds(empty_adapted, 512)
        cache[cache_key] = (empty_qwen_hidden.detach(), empty_adapted.detach())

    empty_qwen_hidden, empty_adapted = cache[cache_key]

    text_backbone = _anima_v3_resolve_text_encoder_backbone(text_encoder)
    qwen_outputs = text_backbone(
        input_ids=qwen_input_ids,
        attention_mask=qwen_attention_mask,
    )
    qwen_hidden = qwen_outputs.last_hidden_state.to(dtype=dtype)
    qwen_hidden = _apply_weights_comfy_anchor(
        qwen_hidden,
        empty_qwen_hidden,
        qwen_weight_tensor,
        strength=qwen_weight_strength,
        clamp_min=weight_clamp_min,
        clamp_max=weight_clamp_max,
    )

    adapted = llm_adapter(
        source_hidden_states=qwen_hidden,
        target_input_ids=t5_input_ids,
    ).to(dtype=dtype)
    adapted = _pad_or_crop_anima_embeds(adapted, 512)
    adapted = _apply_weights_comfy_anchor(
        adapted,
        empty_adapted,
        t5_weight_tensor,
        strength=adapter_weight_strength,
        clamp_min=weight_clamp_min,
        clamp_max=weight_clamp_max,
    )

    if int(num_images_per_prompt) > 1:
        adapted = adapted.repeat_interleave(int(num_images_per_prompt), dim=0)
    return adapted


@torch.no_grad()
def _anima_encode_final_weighted(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: float = 0.0,
    weight_clamp_max: float = 3.0,
) -> torch.Tensor:
    """Fallback for prompt_tokenizer-only AnimaPipeline builds.

    This path obtains the final Anima conditioning from pipe.encode_prompt(clean_text),
    then applies ComfyUI-style token weighting directly to the final conditioning.
    It avoids pipe.tokenizer and llm_adapter entirely.
    """
    tokenizer = _anima_get_component(pipe, ["tokenizer", "prompt_tokenizer"], required=True)
    clean_text, _ = _strip_attention_syntax(text or "")

    base = _anima_encode_cond_only(
        pipe,
        clean_text,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # If there is no actual weighting syntax, return the pipeline-native result.
    parsed = parse_prompt_attention(text or "")
    if not any(abs(float(w) - 1.0) > 1e-6 for _, w in parsed):
        return base

    cache = getattr(pipe, "_sd_embed_anima_empty_cache", None)
    if cache is None:
        cache = {}
        setattr(pipe, "_sd_embed_anima_empty_cache", cache)

    cache_key = ("final", id(tokenizer), str(base.device), str(base.dtype), int(num_images_per_prompt), int(max_sequence_length), tuple(base.shape))
    if cache_key not in cache:
        empty = _anima_encode_cond_only(
            pipe,
            "",
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        ).to(device=base.device, dtype=base.dtype)
        cache[cache_key] = empty.detach()
    empty = cache[cache_key]

    # Build token weights from the syntax-stripped prompt.  Use the output sequence
    # length as max length so the weight tensor aligns with prompt_embeds.
    seq_len = int(base.shape[1])
    _ids, weights, _mask, _ = _token_weights_from_offsets(
        tokenizer,
        text or "",
        max_sequence_length=seq_len,
    )
    weight_tensor = torch.tensor([weights], dtype=base.dtype, device=base.device)

    return _apply_weights_comfy_anchor(
        base,
        empty,
        weight_tensor,
        strength=adapter_weight_strength,
        clamp_min=weight_clamp_min,
        clamp_max=weight_clamp_max,
    )


@torch.no_grad()
def _anima_encode_weighted_single(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: float = 0.0,
    weight_clamp_max: float = 3.0,
) -> torch.Tensor:
    direct = _anima_encode_direct_if_possible(
        pipe,
        text,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
    )
    if direct is not None:
        return direct

    return _anima_encode_final_weighted(
        pipe,
        text,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
    )


@torch.no_grad()
def _mix_AND_anima(
    pipe,
    text: str,
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: float = 0.0,
    weight_clamp_max: float = 3.0,
    and_strength: float = 0.60,
    base_bias: float = 4.0,
) -> torch.Tensor:
    parts = _split_top_level_AND(text)
    parsed: List[Tuple[str, float]] = []
    for p in parts:
        t, w = _split_suffix_weight_top_level(p)
        t = (t or "").strip()
        if t:
            parsed.append((t, float(w)))

    if len(parsed) <= 1:
        return _anima_encode_weighted_single(
            pipe,
            text,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )

    clean_all, _ = _strip_attention_syntax(text)
    base = _anima_encode_weighted_single(
        pipe,
        clean_all,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
    )

    mixed = base * float(base_bias)
    denom = abs(float(base_bias)) + 1e-6
    for t, w in parsed:
        ci = _anima_encode_weighted_single(
            pipe,
            t,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        wf = float(w) if float(w) >= 0.0 else (1.0 + float(w))
        wf = float(max(-2.0, min(3.0, wf)))
        mixed = mixed + ci * wf
        denom = denom + abs(wf)

    mixed = mixed / denom
    return base + (mixed - base) * float(and_strength)




def _anima_format_weight_value(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "1"


def _anima_semantic_make_frontend(
    pipe,
    *,
    mode: str,
    target_t5_tokens: int,
    qwen_input_max_tokens: int,
    compiler_max_new_tokens: int,
    system_prompt: Optional[str],
    tag_resolver: Optional[Any],
    tag_resolver_path: Optional[Union[str, Path]],
    process_negative_prompt: bool,
    allow_generation: bool,
    generation_kwargs: Optional[Dict[str, Any]],
):
    if AnimaSemanticPromptFrontend is None:
        raise ImportError(
            "anima_semantic_prompt.py is required for enable_semantic=True. "
            "Place it next to embedding_funcs.py or install it on PYTHONPATH."
        )
    resolver_obj = tag_resolver
    if resolver_obj is None and tag_resolver_path:
        if TagLexiconResolver is None:
            raise ImportError("TagLexiconResolver is unavailable; cannot load semantic tag resolver JSON.")
        resolver_obj = TagLexiconResolver.from_json(Path(tag_resolver_path))
    kwargs: Dict[str, Any] = {
        "mode": mode,
        "target_t5_tokens": int(target_t5_tokens),
        "qwen_input_max_tokens": int(qwen_input_max_tokens),
        "compiler_max_new_tokens": int(compiler_max_new_tokens),
        "tag_resolver": resolver_obj,
        "process_negative_prompt": bool(process_negative_prompt),
        "allow_generation": bool(allow_generation),
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = str(system_prompt)
    if generation_kwargs is not None:
        kwargs["generation_kwargs"] = dict(generation_kwargs)
    return AnimaSemanticPromptFrontend(pipe, **kwargs)


def _anima_semantic_has_attention_markup(text: str) -> bool:
    s = str(text or "")
    return any(ch in s for ch in ("(", ")", "[", "]", "{", "}"))


def _anima_semantic_extract_uniform_attention(text: str) -> Optional[Tuple[str, float]]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        parts = parse_prompt_attention(s)
    except Exception:
        return None
    normalized_parts = [(str(piece or ""), float(weight)) for piece, weight in parts if str(piece or "")]
    if not normalized_parts:
        return None
    weights = {round(weight, 6) for _piece, weight in normalized_parts}
    if len(weights) != 1:
        return None
    merged = "".join(piece for piece, _weight in normalized_parts).strip()
    if not merged:
        return None
    return merged, float(normalized_parts[0][1])


def _anima_semantic_compile_text(frontend, text: str, *, negative: bool) -> str:
    result = frontend.process_one(text, negative=negative)
    return result.compiled if hasattr(result, "compiled") else str(result)


def _anima_semantic_compile_item(frontend, item: str, *, negative: bool) -> str:
    stripped = str(item or "").strip()
    if not stripped:
        return ""
    if stripped.upper() == "BREAK":
        return "BREAK"

    uniform = _anima_semantic_extract_uniform_attention(stripped)
    if uniform is not None:
        core, weight = uniform
        compiled = _anima_semantic_compile_text(frontend, core, negative=negative).strip()
        if abs(float(weight) - 1.0) < 1e-6:
            return compiled
        return f"({compiled}:{_anima_format_weight_value(weight)})"

    # Mixed inline weighting inside one comma item is difficult to rewrite safely.
    # Preserve the original item rather than risking weight / scope corruption.
    if _anima_semantic_has_attention_markup(stripped):
        return stripped

    return _anima_semantic_compile_text(frontend, stripped, negative=negative).strip()


def _anima_semantic_compile_section(frontend, text: str, *, negative: bool) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""

    # Fast path: no attention syntax in the whole section, so let the semantic
    # compiler see the full clause for better relation understanding.
    if not _anima_semantic_has_attention_markup(stripped):
        return _anima_semantic_compile_text(frontend, stripped, negative=negative).strip()

    items = _anima_v3_split_top_level_comma_items(stripped)
    compiled_items: List[str] = []
    for item in items:
        compiled = _anima_semantic_compile_item(frontend, item, negative=negative).strip()
        if compiled:
            compiled_items.append(compiled)
    return ", ".join(compiled_items)


def _anima_semantic_compile_break_aware(frontend, text: str, *, negative: bool) -> str:
    items = _anima_v3_split_top_level_comma_items(str(text or ""))
    if not items:
        return ""
    sections: List[str] = []
    current_items: List[str] = []

    def flush_current() -> None:
        nonlocal current_items
        section_text = ", ".join([it.strip() for it in current_items if str(it).strip()])
        compiled = _anima_semantic_compile_section(frontend, section_text, negative=negative).strip()
        if compiled:
            sections.append(compiled)
        current_items = []

    saw_break = False
    for item in items:
        stripped = str(item or "").strip()
        if not stripped:
            continue
        if stripped.upper() == "BREAK":
            flush_current()
            sections.append("BREAK")
            saw_break = True
            continue
        current_items.append(stripped)
    flush_current()
    if not saw_break:
        return sections[0] if len(sections) == 1 else ", ".join(sections)
    out_items: List[str] = []
    for section in sections:
        if section == "BREAK":
            if out_items and out_items[-1] != "BREAK":
                out_items.append("BREAK")
            continue
        out_items.append(section)
    return ", ".join(out_items).strip(", ")


def _anima_semantic_compile_prompt(frontend, text: str, *, negative: bool) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    if _has_top_level_AND(raw):
        compiled_segments: List[str] = []
        for segment in _split_top_level_AND(raw):
            segment_text, segment_weight = _split_suffix_weight_top_level(segment)
            compiled = _anima_semantic_compile_break_aware(frontend, segment_text, negative=negative).strip()
            if not compiled:
                compiled = str(segment_text or "").strip()
            if abs(float(segment_weight) - 1.0) >= 1e-6:
                compiled = f"{compiled}:{_anima_format_weight_value(segment_weight)}"
            compiled_segments.append(compiled)
        return " AND ".join(seg for seg in compiled_segments if str(seg).strip())
    return _anima_semantic_compile_break_aware(frontend, raw, negative=negative)


@torch.no_grad()
def get_weighted_text_embeddings_anima(
    pipe,
    prompt: Union[str, List[str]] = "",
    neg_prompt: Union[str, List[str]] = "",
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: float = 0.0,
    weight_clamp_max: float = 3.0,
    enable_AND: bool = True,
    and_strength: float = 0.60,
    base_bias: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted embeddings for Anima.

    Supports both known Anima pipeline layouts:
      1. tokenizer + t5_tokenizer + llm_adapter
      2. prompt_tokenizer + encode_prompt, with the adapter hidden inside the pipeline

    The second layout fixes: AttributeError: 'AnimaPipeline' object has no attribute 'tokenizer'.
    """
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    try:
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = list(prompt)

        if isinstance(neg_prompt, str):
            neg_list = [neg_prompt] * len(prompt_list)
        else:
            neg_list = list(neg_prompt)
            if len(neg_list) != len(prompt_list):
                raise ValueError(
                    f"The number of prompts and neg_prompts not matched: {len(prompt_list)} / {len(neg_list)}"
                )

        def _encode(text: str) -> torch.Tensor:
            if enable_AND and _has_top_level_AND(text or ""):
                return _mix_AND_anima(
                    pipe,
                    text or "",
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    qwen_weight_strength=qwen_weight_strength,
                    adapter_weight_strength=adapter_weight_strength,
                    weight_clamp_min=weight_clamp_min,
                    weight_clamp_max=weight_clamp_max,
                    and_strength=and_strength,
                    base_bias=base_bias,
                )
            return _anima_encode_weighted_single(
                pipe,
                text or "",
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                qwen_weight_strength=qwen_weight_strength,
                adapter_weight_strength=adapter_weight_strength,
                weight_clamp_min=weight_clamp_min,
                weight_clamp_max=weight_clamp_max,
            )

        pos_list = [_encode(p) for p in prompt_list]
        neg_out = [_encode(n) for n in neg_list]
        return torch.cat(pos_list, dim=0), torch.cat(neg_out, dim=0)
    finally:
        dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)

# ============================================================
# Anima (AM) embeddings - diffusers_anima compatible override
# ============================================================
# This override is intentionally placed after the earlier Anima helpers so that
# get_weighted_text_embeddings_anima resolves to this implementation.
# It follows diffusers_anima.text_encoding:
#   prompt_tokenizer -> text_encoder(Qwen hidden) -> transformer.preprocess_text_embeds
# and injects sd_embed / A1111 / ComfyUI-style prompt weights before the final
# Anima conditioning tensor is built.

try:
    from contextlib import contextmanager as _sd_embed_contextmanager
except Exception:  # pragma: no cover
    _sd_embed_contextmanager = None

_ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID = 151643
_ANIMA_CONDITIONING_MAX_LENGTH = 512

# Anima's native helper commonly produces a 512-token conditioning window.
# The CLIP/SDXL-style long-prompt strategy implemented here preserves later
# windows by concatenating native per-window conditions along the sequence
# dimension instead of compressing them back into a single 512-token tensor.
# Older fusion modes are still kept as fallbacks/comparison strategies.
_ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT = "chunk_concat"
_ANIMA_LONG_PROMPT_FUSION_CHUNK_BLEND = "chunk_blend"
_ANIMA_LONG_PROMPT_FUSION_CHUNK_SLOTS = "chunk_slots"
_ANIMA_LONG_PROMPT_FUSION_CHUNK_RESIDUAL = "chunk_residual"
_ANIMA_LONG_PROMPT_FUSION_TRUNCATE = "truncate"


def _anima_v3_flatten_ids(value) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, list):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in (value or [])]


def _anima_v3_flatten_offsets(value) -> List[Tuple[int, int]]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, list):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    out: List[Tuple[int, int]] = []
    for item in value or []:
        try:
            out.append((int(item[0]), int(item[1])))
        except Exception:
            out.append((0, 0))
    return out


def _anima_v3_weighted_spans(prompt: str) -> Tuple[str, List[Tuple[int, int, float]], bool]:
    segs = parse_prompt_attention(prompt or "")
    clean_parts: List[str] = []
    spans: List[Tuple[int, int, float]] = []
    pos = 0
    has_weight = False
    for text, weight in segs:
        text = text or ""
        weight = float(weight)
        clean_parts.append(text)
        if len(text) > 0:
            spans.append((pos, pos + len(text), weight))
            if abs(weight - 1.0) > 1e-6:
                has_weight = True
        pos += len(text)
    return "".join(clean_parts), spans, has_weight


def _anima_v3_weights_from_offsets(
    offsets: List[Tuple[int, int]],
    spans: List[Tuple[int, int, float]],
) -> List[float]:
    if not offsets:
        return []
    weights: List[float] = []
    span_i = 0
    for ts, te in offsets:
        if te <= ts:
            weights.append(1.0)
            continue
        while span_i < len(spans) and spans[span_i][1] <= ts:
            span_i += 1
        weighted = 0.0
        count = 0
        j = span_i
        while j < len(spans) and spans[j][0] < te:
            s0, s1, weight = spans[j]
            overlap = max(0, min(te, s1) - max(ts, s0))
            if overlap > 0:
                weighted += float(weight) * overlap
                count += overlap
            j += 1
        weights.append(weighted / count if count > 0 else 1.0)
    return weights


def _anima_v3_tokenize_segment_fallback(tokenizer, segs: List[Tuple[str, float]]) -> Tuple[List[int], List[float]]:
    token_ids: List[int] = []
    token_weights: List[float] = []
    for text, weight in segs:
        text = text or ""
        if not text:
            continue
        encoded = tokenizer(
            [text],
            add_special_tokens=False,
            truncation=False,
            return_tensors=None,
        )
        ids = _anima_v3_flatten_ids(getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else []))
        token_ids.extend(ids)
        token_weights.extend([float(weight)] * len(ids))
    return token_ids, token_weights


def _anima_v3_tokenize_with_weights(
    tokenizer,
    prompt: str,
    *,
    empty_token_id: int,
    add_eos: bool = False,
    eos_token_id: Optional[int] = None,
    truncate_to: Optional[int] = None,
) -> Tuple[List[int], List[float], str, bool]:
    """Tokenize a weighted prompt while preserving diffusers_anima token policy.

    For fast tokenizers, this tokenizes the full syntax-stripped text once and maps
    weights by offset_mapping. This keeps token boundaries closest to the native
    AnimaPromptTokenizer. Slow tokenizers fall back to per-segment tokenization.
    """
    clean_text, spans, has_weight = _anima_v3_weighted_spans(prompt or "")
    if clean_text:
        try:
            try:
                encoded = tokenizer(
                    [clean_text],
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                    return_tensors=None,
                    verbose=False,
                )
            except TypeError:
                encoded = tokenizer(
                    [clean_text],
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                    return_tensors=None,
                )
            ids = _anima_v3_flatten_ids(getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else []))
            offsets = _anima_v3_flatten_offsets(getattr(encoded, "offset_mapping", encoded.get("offset_mapping") if isinstance(encoded, dict) else []))
            weights = _anima_v3_weights_from_offsets(offsets, spans)
            if len(weights) != len(ids):
                raise RuntimeError("offset/ids length mismatch")
        except Exception:
            segs = [(text, float(weight)) for text, weight in parse_prompt_attention(prompt or "")]
            ids, weights = _anima_v3_tokenize_segment_fallback(tokenizer, segs)
    else:
        ids, weights = [], []

    if not ids:
        ids = [int(empty_token_id)]
        weights = [1.0]

    if add_eos:
        if eos_token_id is None:
            eos_token_id = 1
        if int(ids[-1]) != int(eos_token_id):
            ids = [*ids, int(eos_token_id)]
            weights = [*weights, 1.0]

    if truncate_to is not None and int(truncate_to) > 0:
        ids = ids[: int(truncate_to)]
        weights = weights[: int(truncate_to)]
        if add_eos and eos_token_id is not None and ids:
            # Keep the T5 sequence terminated even when truncation cut the eos.
            ids[-1] = int(eos_token_id)
            weights[-1] = 1.0

    return ids, [float(w) for w in weights], clean_text, bool(has_weight)


def _anima_v3_scale_weight_tensor(
    weights: torch.Tensor,
    *,
    strength: float,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> torch.Tensor:
    factor = _negpip_factor(weights)
    factor = 1.0 + (factor - 1.0) * float(strength)
    if clamp_min is not None or clamp_max is not None:
        factor = factor.clamp(
            min=float(clamp_min) if clamp_min is not None else None,
            max=float(clamp_max) if clamp_max is not None else None,
        )
    return factor


def _anima_v3_apply_qwen_weights(
    qwen_hidden: torch.Tensor,
    qwen_weights: torch.Tensor,
    qwen_mask: torch.Tensor,
    *,
    strength: float,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> torch.Tensor:
    if qwen_hidden.ndim != 3:
        raise ValueError(f"expected qwen_hidden as (B,T,D), got {tuple(qwen_hidden.shape)}")
    if qwen_weights.ndim == 2:
        qwen_weights = qwen_weights.unsqueeze(-1)
    if qwen_mask.ndim == 2:
        mask = qwen_mask.to(device=qwen_hidden.device, dtype=qwen_hidden.dtype).unsqueeze(-1)
    else:
        mask = qwen_mask.to(device=qwen_hidden.device, dtype=qwen_hidden.dtype)

    factor = _anima_v3_scale_weight_tensor(
        qwen_weights.to(device=qwen_hidden.device, dtype=qwen_hidden.dtype),
        strength=strength,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    # Padding positions should stay unchanged.
    factor = torch.where(mask > 0, factor, torch.ones_like(factor))

    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    anchor = (qwen_hidden * mask).sum(dim=1, keepdim=True) / denom
    return anchor + (qwen_hidden - anchor) * factor


if _sd_embed_contextmanager is not None:
    @_sd_embed_contextmanager
    def _anima_v3_module_execution_context(
        module: torch.nn.Module,
        *,
        execution_device: str,
        execution_dtype: torch.dtype,
        enable_offload: bool,
    ):
        if enable_offload and str(execution_device) != "cpu":
            module.to(device=execution_device, dtype=execution_dtype)
            try:
                yield
            finally:
                module.to(device="cpu")
                if str(execution_device).startswith("cuda"):
                    torch.cuda.empty_cache()
            return
        yield
else:  # pragma: no cover
    def _anima_v3_module_execution_context(*args, **kwargs):
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return _NullCtx()


def _anima_v3_pipeline_runtime(pipe) -> Tuple[str, torch.dtype, torch.dtype, bool]:
    execution_device = getattr(pipe, "execution_device", None)
    if execution_device is None:
        execution_device = _anima_get_device(pipe)
    execution_device = str(execution_device)

    model_dtype = getattr(pipe, "model_dtype", None)
    if model_dtype is None:
        model_dtype = _anima_get_dtype(pipe)
    text_encoder_dtype = getattr(pipe, "text_encoder_dtype", None)
    if text_encoder_dtype is None:
        text_encoder_dtype = model_dtype
    enable_offload = bool(getattr(pipe, "use_module_cpu_offload", False))
    return execution_device, model_dtype, text_encoder_dtype, enable_offload


def _anima_v3_get_prompt_tokenizer(pipe):
    prompt_tokenizer = _anima_get_component(pipe, ["prompt_tokenizer"], required=False)
    if prompt_tokenizer is not None:
        qwen_tokenizer = getattr(prompt_tokenizer, "qwen_tokenizer", None)
        t5_tokenizer = getattr(prompt_tokenizer, "t5_tokenizer", None)
        if qwen_tokenizer is None or t5_tokenizer is None:
            raise AttributeError("pipe.prompt_tokenizer must expose qwen_tokenizer and t5_tokenizer")
        return prompt_tokenizer, qwen_tokenizer, t5_tokenizer

    # Compatibility with older experimental layouts.
    qwen_tokenizer = _anima_get_component(pipe, ["tokenizer", "qwen_tokenizer"], required=True)
    t5_tokenizer = _anima_get_component(pipe, ["t5_tokenizer", "tokenizer_2", "target_tokenizer"], required=True)

    class _PromptTokenizerShim:
        pass

    prompt_tokenizer = _PromptTokenizerShim()
    prompt_tokenizer.qwen_tokenizer = qwen_tokenizer
    prompt_tokenizer.t5_tokenizer = t5_tokenizer
    return prompt_tokenizer, qwen_tokenizer, t5_tokenizer


@torch.no_grad()
def _anima_v3_prepare_condition_inputs(
    pipe,
    prompts: List[str],
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    t5_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(prompts) == 0:
        raise ValueError("prompt batch must not be empty")

    prompt_tokenizer, qwen_tokenizer, t5_tokenizer = _anima_v3_get_prompt_tokenizer(pipe)
    text_encoder = _anima_get_component(pipe, ["text_encoder"], required=True)
    execution_device, model_dtype, text_encoder_dtype, enable_offload = _anima_v3_pipeline_runtime(pipe)

    qwen_pad = getattr(qwen_tokenizer, "pad_token_id", None)
    if qwen_pad is None:
        qwen_pad = _ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_pad = getattr(t5_tokenizer, "pad_token_id", None)
    if t5_pad is None:
        t5_pad = 0
    t5_eos = getattr(t5_tokenizer, "eos_token_id", None)
    if t5_eos is None:
        t5_eos = 1

    qwen_batches: List[List[int]] = []
    qwen_weight_batches: List[List[float]] = []
    t5_batches: List[List[int]] = []
    t5_weight_batches: List[List[float]] = []
    max_qwen_len = 0
    max_t5_len = 0

    # Leave at least one slot for T5 eos if truncating.
    qwen_trunc = int(max_sequence_length) if max_sequence_length and max_sequence_length > 0 else None
    t5_trunc = int(max_sequence_length) if max_sequence_length and max_sequence_length > 0 else None

    for text in prompts:
        q_ids, q_weights, _clean_q, _has_q = _anima_v3_tokenize_with_weights(
            qwen_tokenizer,
            text or "",
            empty_token_id=int(qwen_pad),
            add_eos=False,
            eos_token_id=None,
            truncate_to=qwen_trunc,
        )
        t_ids, t_weights, _clean_t, _has_t = _anima_v3_tokenize_with_weights(
            t5_tokenizer,
            text or "",
            empty_token_id=int(t5_eos),
            add_eos=True,
            eos_token_id=int(t5_eos),
            truncate_to=t5_trunc,
        )
        qwen_batches.append(q_ids)
        qwen_weight_batches.append(q_weights)
        t5_batches.append(t_ids)
        t5_weight_batches.append(t_weights)
        max_qwen_len = max(max_qwen_len, len(q_ids))
        max_t5_len = max(max_t5_len, len(t_ids))

    batch_size = len(prompts)
    qwen_ids = torch.full(
        (batch_size, max_qwen_len),
        int(qwen_pad),
        dtype=torch.long,
        device=execution_device,
    )
    qwen_mask = torch.zeros(
        (batch_size, max_qwen_len),
        dtype=torch.long,
        device=execution_device,
    )
    qwen_weights_t = torch.ones(
        (batch_size, max_qwen_len),
        dtype=torch.float32,
        device=execution_device,
    )
    t5_ids = torch.full(
        (batch_size, max_t5_len),
        int(t5_pad),
        dtype=torch.int32,
        device=execution_device,
    )
    t5_weights_t = torch.zeros(
        (batch_size, max_t5_len, 1),
        dtype=torch.float32,
        device=execution_device,
    )

    for i, (q_ids, q_weights, t_ids, t_weights) in enumerate(zip(qwen_batches, qwen_weight_batches, t5_batches, t5_weight_batches)):
        q_len = len(q_ids)
        t_len = len(t_ids)
        qwen_ids[i, :q_len] = torch.tensor(q_ids, dtype=torch.long, device=execution_device)
        qwen_mask[i, :q_len] = 1
        qwen_weights_t[i, :q_len] = torch.tensor(q_weights, dtype=torch.float32, device=execution_device)
        t5_ids[i, :t_len] = torch.tensor(t_ids, dtype=torch.int32, device=execution_device)
        raw_t5_weights = torch.tensor(t_weights, dtype=torch.float32, device=execution_device).view(t_len, 1)
        scaled_t5_weights = _anima_v3_scale_weight_tensor(
            raw_t5_weights,
            strength=t5_weight_strength,
            clamp_min=weight_clamp_min,
            clamp_max=weight_clamp_max,
        )
        t5_weights_t[i, :t_len, :] = scaled_t5_weights

    with _anima_v3_module_execution_context(
        text_encoder,
        execution_device=execution_device,
        execution_dtype=text_encoder_dtype,
        enable_offload=enable_offload,
    ):
        with torch.inference_mode():
            text_backbone = _anima_v3_resolve_text_encoder_backbone(text_encoder)
            out = text_backbone(input_ids=qwen_ids, attention_mask=qwen_mask)
            if isinstance(out, tuple):
                qwen_hidden = out[0]
            else:
                qwen_hidden = out.last_hidden_state
            qwen_hidden = qwen_hidden.to(device=execution_device, dtype=model_dtype)
            qwen_hidden = _anima_v3_apply_qwen_weights(
                qwen_hidden,
                qwen_weights_t,
                qwen_mask,
                strength=qwen_weight_strength,
                clamp_min=weight_clamp_min,
                clamp_max=weight_clamp_max,
            )

    return qwen_hidden, t5_ids, t5_weights_t


def _anima_v3_long_prompt_chunk_size(max_sequence_length: int, long_prompt_chunk_size: Optional[int]) -> int:
    if long_prompt_chunk_size is not None and int(long_prompt_chunk_size) > 0:
        return max(1, min(int(long_prompt_chunk_size), _ANIMA_CONDITIONING_MAX_LENGTH))
    if max_sequence_length is not None and int(max_sequence_length) > 0:
        # Never use a window wider than the final Anima conditioning width.
        # Otherwise preprocess_text_embeds can create information that is then
        # cropped back to 512, reproducing the original weak-tail behavior.
        return max(1, min(int(max_sequence_length), _ANIMA_CONDITIONING_MAX_LENGTH))
    return _ANIMA_CONDITIONING_MAX_LENGTH


def _anima_v3_split_ids_weights(
    token_ids: List[int],
    token_weights: List[float],
    *,
    chunk_size: int,
    empty_token_id: int,
    add_eos: bool = False,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    chunk_size = max(1, int(chunk_size))
    token_ids = [int(x) for x in (token_ids or [])]
    token_weights = [float(x) for x in (token_weights or [])]
    if len(token_weights) < len(token_ids):
        token_weights = [*token_weights, *([1.0] * (len(token_ids) - len(token_weights)))]
    elif len(token_weights) > len(token_ids):
        token_weights = token_weights[:len(token_ids)]

    if not token_ids:
        if add_eos:
            eos = int(eos_token_id if eos_token_id is not None else empty_token_id)
            return [[eos]], [[1.0]]
        return [[int(empty_token_id)]], [[1.0]]

    if add_eos:
        eos = int(eos_token_id if eos_token_id is not None else empty_token_id)
        # Keep one slot for EOS in every T5 window.  This gives each window a
        # complete prompt-like sequence instead of only the final window being
        # terminated.
        content_budget = max(1, chunk_size - 1)
    else:
        eos = None
        content_budget = chunk_size

    id_chunks: List[List[int]] = []
    weight_chunks: List[List[float]] = []
    for start in range(0, len(token_ids), content_budget):
        ids = token_ids[start:start + content_budget]
        weights = token_weights[start:start + content_budget]
        if add_eos:
            ids = [*ids, int(eos)]
            weights = [*weights, 1.0]
        id_chunks.append(ids)
        weight_chunks.append(weights)
    return id_chunks, weight_chunks


def _anima_v3_empty_chunk(*, empty_token_id: int, add_eos: bool = False, eos_token_id: Optional[int] = None) -> Tuple[List[int], List[float]]:
    if add_eos:
        return [int(eos_token_id if eos_token_id is not None else empty_token_id)], [1.0]
    return [int(empty_token_id)], [1.0]


@torch.no_grad()
def _anima_v3_prepare_condition_inputs_from_token_batches(
    pipe,
    *,
    qwen_batches: List[List[int]],
    qwen_weight_batches: List[List[float]],
    t5_batches: List[List[int]],
    t5_weight_batches: List[List[float]],
    qwen_weight_strength: float,
    t5_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(qwen_batches) == 0:
        raise ValueError("token batch must not be empty")
    if not (len(qwen_batches) == len(qwen_weight_batches) == len(t5_batches) == len(t5_weight_batches)):
        raise ValueError("qwen/t5 token batch size mismatch")

    _prompt_tokenizer, qwen_tokenizer, t5_tokenizer = _anima_v3_get_prompt_tokenizer(pipe)
    text_encoder = _anima_get_component(pipe, ["text_encoder"], required=True)
    execution_device, model_dtype, text_encoder_dtype, enable_offload = _anima_v3_pipeline_runtime(pipe)

    qwen_pad = getattr(qwen_tokenizer, "pad_token_id", None)
    if qwen_pad is None:
        qwen_pad = _ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_pad = getattr(t5_tokenizer, "pad_token_id", None)
    if t5_pad is None:
        t5_pad = 0

    max_qwen_len = max(1, max(len(x) for x in qwen_batches))
    max_t5_len = max(1, max(len(x) for x in t5_batches))
    batch_size = len(qwen_batches)

    qwen_ids = torch.full(
        (batch_size, max_qwen_len),
        int(qwen_pad),
        dtype=torch.long,
        device=execution_device,
    )
    qwen_mask = torch.zeros(
        (batch_size, max_qwen_len),
        dtype=torch.long,
        device=execution_device,
    )
    qwen_weights_t = torch.ones(
        (batch_size, max_qwen_len),
        dtype=torch.float32,
        device=execution_device,
    )
    t5_ids = torch.full(
        (batch_size, max_t5_len),
        int(t5_pad),
        dtype=torch.int32,
        device=execution_device,
    )
    t5_weights_t = torch.zeros(
        (batch_size, max_t5_len, 1),
        dtype=torch.float32,
        device=execution_device,
    )

    for i, (q_ids, q_weights, tt_ids, tt_weights) in enumerate(zip(qwen_batches, qwen_weight_batches, t5_batches, t5_weight_batches)):
        q_ids = [int(x) for x in (q_ids or [qwen_pad])]
        tt_ids = [int(x) for x in (tt_ids or [t5_pad])]
        q_weights = [float(x) for x in (q_weights or [1.0] * len(q_ids))]
        tt_weights = [float(x) for x in (tt_weights or [1.0] * len(tt_ids))]
        if len(q_weights) < len(q_ids):
            q_weights = [*q_weights, *([1.0] * (len(q_ids) - len(q_weights)))]
        elif len(q_weights) > len(q_ids):
            q_weights = q_weights[:len(q_ids)]
        if len(tt_weights) < len(tt_ids):
            tt_weights = [*tt_weights, *([1.0] * (len(tt_ids) - len(tt_weights)))]
        elif len(tt_weights) > len(tt_ids):
            tt_weights = tt_weights[:len(tt_ids)]
        q_len = len(q_ids)
        t_len = len(tt_ids)
        qwen_ids[i, :q_len] = torch.tensor(q_ids, dtype=torch.long, device=execution_device)
        qwen_mask[i, :q_len] = 1
        qwen_weights_t[i, :q_len] = torch.tensor(q_weights[:q_len], dtype=torch.float32, device=execution_device)
        t5_ids[i, :t_len] = torch.tensor(tt_ids, dtype=torch.int32, device=execution_device)
        raw_t5_weights = torch.tensor(tt_weights[:t_len], dtype=torch.float32, device=execution_device).view(t_len, 1)
        scaled_t5_weights = _anima_v3_scale_weight_tensor(
            raw_t5_weights,
            strength=t5_weight_strength,
            clamp_min=weight_clamp_min,
            clamp_max=weight_clamp_max,
        )
        t5_weights_t[i, :t_len, :] = scaled_t5_weights

    with _anima_v3_module_execution_context(
        text_encoder,
        execution_device=execution_device,
        execution_dtype=text_encoder_dtype,
        enable_offload=enable_offload,
    ):
        with torch.inference_mode():
            text_backbone = _anima_v3_resolve_text_encoder_backbone(text_encoder)
            out = text_backbone(input_ids=qwen_ids, attention_mask=qwen_mask)
            if isinstance(out, tuple):
                qwen_hidden = out[0]
            else:
                qwen_hidden = out.last_hidden_state
            qwen_hidden = qwen_hidden.to(device=execution_device, dtype=model_dtype)
            qwen_hidden = _anima_v3_apply_qwen_weights(
                qwen_hidden,
                qwen_weights_t,
                qwen_mask,
                strength=qwen_weight_strength,
                clamp_min=weight_clamp_min,
                clamp_max=weight_clamp_max,
            )

    return qwen_hidden, t5_ids, t5_weights_t


def _anima_v3_count_tokens_no_special(tokenizer, text: str) -> int:
    try:
        encoded = tokenizer(text or "", add_special_tokens=False, truncation=False)
    except TypeError:
        encoded = tokenizer(text or "", truncation=False)
    ids = _anima_v3_flatten_ids(getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else []))
    return len(ids)


def _anima_v3_escape_weighted_text(text: str) -> str:
    # Keep the reconstructed prompt parseable by parse_prompt_attention.
    return (text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")


def _anima_v3_serialize_weighted_segments(segments: List[Tuple[str, float]]) -> str:
    out: List[str] = []
    for t, w in segments:
        if not t:
            continue
        wf = float(w)
        if abs(wf - 1.0) < 1e-6:
            out.append(t)
        else:
            out.append(f"({_anima_v3_escape_weighted_text(t)}:{wf:.6g})")
    return "".join(out)


def _anima_v3_split_text_piece(piece: str) -> List[str]:
    # Prompts are usually comma-tag based.  Prefer comma boundaries, then word
    # boundaries, and finally character chunks for pathological long text.
    piece = piece or ""
    if not piece:
        return []
    comma_parts: List[str] = []
    parts = piece.split(",")
    for i, part in enumerate(parts):
        suffix = "," if i < len(parts) - 1 else ""
        comma_parts.append(part + suffix)
    out: List[str] = []
    for part in comma_parts:
        if len(part) <= 256:
            out.append(part)
            continue
        words = part.split(" ")
        if len(words) > 1:
            for j, word in enumerate(words):
                out.append(word + (" " if j < len(words) - 1 else ""))
        else:
            for j in range(0, len(part), 96):
                out.append(part[j:j + 96])
    return [x for x in out if x]


def _anima_v3_split_prompt_text_dual_budget(
    qwen_tokenizer,
    t5_tokenizer,
    text: str,
    *,
    chunk_size: int,
) -> List[str]:
    """Split the same weighted text into windows that fit both Qwen and T5.

    The previous token-id splitter chunked Qwen and T5 independently.  That can
    pair Qwen chunk N with a semantically different T5 chunk N, which is a very
    common cause of long-prompt dropouts.  This splitter keeps both encoders on
    the same text window.
    """
    chunk_size = max(2, int(chunk_size))
    t5_budget = max(1, chunk_size - 1)  # one slot for EOS
    segs = parse_prompt_attention(text or "")
    if not segs:
        segs = [("", 1.0)]

    chunks: List[str] = []
    current: List[Tuple[str, float]] = []

    def fits(candidate: List[Tuple[str, float]]) -> bool:
        serialized = _anima_v3_serialize_weighted_segments(candidate)
        clean, _spans, _has_weight = _anima_v3_weighted_spans(serialized)
        q_len = _anima_v3_count_tokens_no_special(qwen_tokenizer, clean)
        t_len = _anima_v3_count_tokens_no_special(t5_tokenizer, clean)
        return q_len <= chunk_size and t_len <= t5_budget

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(_anima_v3_serialize_weighted_segments(current))
            current = []

    for seg_text, weight in segs:
        for piece in _anima_v3_split_text_piece(seg_text):
            candidate = [*current, (piece, float(weight))]
            if fits(candidate):
                current = candidate
                continue
            flush()
            single = [(piece, float(weight))]
            if fits(single):
                current = single
                continue
            # Extremely long token/word fallback: let the tokenizer-level
            # truncation handle this one atomic piece rather than dropping it.
            chunks.append(_anima_v3_serialize_weighted_segments(single))
    flush()
    return chunks or [text or ""]


def _anima_v3_match_condition_stats(x: torch.Tensor, reference: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Match per-sample/channel sequence statistics.  This prevents the long
    # prompt fusion from shifting the global color/style distribution, which is
    # the main reason averaged chunks look washed out or over-tinted.
    if x.shape != reference.shape or x.dim() != 3:
        return x
    x32 = x.float()
    ref32 = reference.float()
    x_mean = x32.mean(dim=1, keepdim=True)
    x_std = x32.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    r_mean = ref32.mean(dim=1, keepdim=True)
    r_std = ref32.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    y = (x32 - x_mean) / x_std * r_std + r_mean
    return y.to(dtype=x.dtype, device=x.device)


def _anima_v3_pad_condition_to_length(cond: torch.Tensor, target_length: int) -> torch.Tensor:
    if cond.dim() != 3:
        return cond
    cur = int(cond.shape[1])
    tgt = int(target_length)
    if cur == tgt:
        return cond
    if cur > tgt:
        return cond[:, :tgt, :]

    # Match diffusers_anima.text_encoding.build_condition: pad missing
    # conditioning slots with zeros, not repeated EOS/last-token features.
    # Repeating the last token makes long chunks over-emphasise tail content and
    # shifts the global style/color distribution.
    pad_len = tgt - cur
    pad = torch.zeros(
        cond.shape[0],
        pad_len,
        cond.shape[2],
        dtype=cond.dtype,
        device=cond.device,
    )
    return torch.cat([cond, pad], dim=1)


def _anima_v3_pad_condition_list(conditions: List[torch.Tensor], target_length: Optional[int] = None) -> List[torch.Tensor]:
    if not conditions:
        return []
    if target_length is None:
        target_length = max(int(c.shape[1]) for c in conditions)
    return [_anima_v3_pad_condition_to_length(c, int(target_length)) for c in conditions]


def _anima_v3_align_pos_neg_conditions(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pos.dim() != 3 or neg.dim() != 3:
        return pos, neg
    target_length = max(int(pos.shape[1]), int(neg.shape[1]))
    return (
        _anima_v3_pad_condition_to_length(pos, target_length),
        _anima_v3_pad_condition_to_length(neg, target_length),
    )


def _anima_v3_fuse_long_prompt_conditions(
    conditions: List[torch.Tensor],
    *,
    strength: float,
    chunk_decay: float,
    mode: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    anchor_tokens: int = 160,
) -> torch.Tensor:
    if not conditions:
        raise ValueError("conditions must not be empty")
    if len(conditions) == 1:
        return conditions[0]

    base = conditions[0]
    mode = str(mode or _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT).lower()
    s = float(strength)
    if not math.isfinite(s):
        s = 1.0
    s = max(0.0, min(s, 2.0))

    decay = float(chunk_decay)
    if not math.isfinite(decay) or decay <= 0:
        decay = 1.0

    # CLIP/SDXL-style mode: keep each native chunk intact and concatenate them
    # on the sequence axis. This avoids the overwrite/averaging problem that
    # happens when multiple 512-token windows are compressed back into a single
    # conditioning window.
    if mode in (_ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT, "clip_concat", "seq_concat"):
        return torch.cat([c.to(device=base.device, dtype=base.dtype) for c in conditions], dim=1)

    # Legacy behavior kept for comparison/backward compatibility.
    if mode == _ANIMA_LONG_PROMPT_FUSION_CHUNK_BLEND:
        conditions = _anima_v3_pad_condition_list(conditions)
        base = conditions[0]
        weights = torch.tensor(
            [decay ** i for i in range(len(conditions))],
            dtype=torch.float32,
            device=base.device,
        )
        weights = weights / weights.sum().clamp_min(1e-6)
        stacked = torch.stack(conditions, dim=0)
        fused = (stacked * weights.view(-1, 1, 1, 1).to(dtype=stacked.dtype)).sum(dim=0)
        fused = _anima_v3_match_condition_stats(fused, base)
        return base + (fused - base) * s

    # Residual mode: safer than averaging, but still compresses all later
    # chunks into one global residual.
    if mode == _ANIMA_LONG_PROMPT_FUSION_CHUNK_RESIDUAL:
        conditions = _anima_v3_pad_condition_list(conditions)
        base = conditions[0]
        rest = conditions[1:]
        weights = torch.tensor(
            [decay ** i for i in range(len(rest))],
            dtype=torch.float32,
            device=base.device,
        )
        weights = weights / weights.sum().clamp_min(1e-6)
        stacked = torch.stack([_anima_v3_match_condition_stats(c, base) for c in rest], dim=0)
        rest_mean = (stacked * weights.view(-1, 1, 1, 1).to(dtype=stacked.dtype)).sum(dim=0)
        return base + (rest_mean - base) * (0.35 * s)

    # Slot mode: keep the first chunk as the anchor and inject later chunks into
    # tail slots. Still useful as an alternative, but concat is now the default.
    conditions = _anima_v3_pad_condition_list(conditions)
    base = conditions[0]
    fused = base.clone()
    seq_len = int(base.shape[1])
    try:
        anchor = int(anchor_tokens)
    except Exception:
        anchor = 160
    if not math.isfinite(float(anchor)):
        anchor = 160
    anchor = max(0, min(anchor, max(0, seq_len - 1)))
    tail = max(1, seq_len - anchor)
    rest_count = max(1, len(conditions) - 1)
    slot_len = max(1, tail // rest_count)
    blend = max(0.0, min(1.0, 0.85 * s))

    for j, cond in enumerate(conditions[1:]):
        cond = _anima_v3_match_condition_stats(cond.to(device=base.device, dtype=base.dtype), base)
        dst0 = anchor + j * slot_len
        if dst0 >= seq_len:
            break
        dst1 = seq_len if j == len(conditions) - 2 else min(seq_len, dst0 + slot_len)
        n = max(0, dst1 - dst0)
        if n <= 0:
            continue
        src = cond[:, :n, :]
        fused[:, dst0:dst1, :] = fused[:, dst0:dst1, :] * (1.0 - blend) + src * blend

    corrected = _anima_v3_match_condition_stats(fused, base)
    return fused + (corrected - fused) * 0.75


@torch.no_grad()
def _anima_v3_encode_long_single(
    pipe,
    text: str,
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    long_prompt_strategy: str,
    long_prompt_chunk_size: Optional[int],
    long_prompt_chunk_decay: float,
    long_prompt_strength: float,
    long_prompt_anchor_tokens: int,
) -> torch.Tensor:
    _prompt_tokenizer, qwen_tokenizer, t5_tokenizer = _anima_v3_get_prompt_tokenizer(pipe)
    qwen_pad = getattr(qwen_tokenizer, "pad_token_id", None)
    if qwen_pad is None:
        qwen_pad = _ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_eos = getattr(t5_tokenizer, "eos_token_id", None)
    if t5_eos is None:
        t5_eos = 1

    q_ids, _q_weights, _clean_q, _has_q = _anima_v3_tokenize_with_weights(
        qwen_tokenizer,
        text or "",
        empty_token_id=int(qwen_pad),
        add_eos=False,
        eos_token_id=None,
        truncate_to=None,
    )
    t_ids, _t_weights, _clean_t, _has_t = _anima_v3_tokenize_with_weights(
        t5_tokenizer,
        text or "",
        empty_token_id=int(t5_eos),
        add_eos=False,
        eos_token_id=None,
        truncate_to=None,
    )

    chunk_size = _anima_v3_long_prompt_chunk_size(max_sequence_length, long_prompt_chunk_size)

    # Fast path: keep the old/native behavior exactly for prompts that fit in
    # one conditioning window.  T5 needs one extra slot for EOS.
    if len(q_ids) <= chunk_size and (len(t_ids) + 1) <= chunk_size:
        qwen_hidden, t5_ids, t5_weights = _anima_v3_prepare_condition_inputs(
            pipe,
            [text or ""],
            max_sequence_length=chunk_size,
            qwen_weight_strength=qwen_weight_strength,
            t5_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        return _anima_v3_build_condition(
            pipe,
            qwen_hidden=qwen_hidden,
            t5_ids=t5_ids,
            t5_weights=t5_weights,
        )

    text_chunks = _anima_v3_split_prompt_text_dual_budget(
        qwen_tokenizer,
        t5_tokenizer,
        text or "",
        chunk_size=chunk_size,
    )

    chunk_conditions: List[torch.Tensor] = []
    concat_mode = str(long_prompt_strategy or _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT).lower() in (
        _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
        "clip_concat",
        "seq_concat",
    )
    for chunk_text in text_chunks:
        qwen_hidden, t5_ids_t, t5_weights_t = _anima_v3_prepare_condition_inputs(
            pipe,
            [chunk_text or ""],
            max_sequence_length=chunk_size,
            qwen_weight_strength=qwen_weight_strength,
            t5_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        chunk_conditions.append(_anima_v3_build_condition(
            pipe,
            qwen_hidden=qwen_hidden,
            t5_ids=t5_ids_t,
            t5_weights=t5_weights_t,
            # Each long-prompt window should be a native Anima conditioning
            # window.  Keeping 512 slots per chunk makes the returned seq_len
            # reflect the number of recognised windows: 512, 1024, 1536, ...
            target_length=_ANIMA_CONDITIONING_MAX_LENGTH,
        ))

    return _anima_v3_fuse_long_prompt_conditions(
        chunk_conditions,
        strength=long_prompt_strength,
        chunk_decay=long_prompt_chunk_decay,
        mode=long_prompt_strategy,
        anchor_tokens=long_prompt_anchor_tokens,
    )


@torch.no_grad()
def _anima_v3_build_condition(
    pipe,
    *,
    qwen_hidden: torch.Tensor,
    t5_ids: torch.Tensor,
    t5_weights: torch.Tensor,
    target_length: Optional[int] = _ANIMA_CONDITIONING_MAX_LENGTH,
) -> torch.Tensor:
    transformer = _anima_get_component(pipe, ["transformer"], required=True)
    execution_device, model_dtype, _text_encoder_dtype, enable_offload = _anima_v3_pipeline_runtime(pipe)
    if not hasattr(transformer, "preprocess_text_embeds"):
        # Last-resort compatibility.  This path cannot use native T5 weights.
        if hasattr(pipe, "encode_prompt"):
            raise AttributeError(
                "Anima transformer has no preprocess_text_embeds; use pipe.encode_prompt for unweighted prompts, "
                "or update diffusers_anima to expose transformer.preprocess_text_embeds."
            )
        raise AttributeError("Anima transformer is missing preprocess_text_embeds")

    with _anima_v3_module_execution_context(
        transformer,
        execution_device=execution_device,
        execution_dtype=model_dtype,
        enable_offload=enable_offload,
    ):
        with torch.inference_mode():
            cond = transformer.preprocess_text_embeds(
                qwen_hidden.to(device=execution_device, dtype=model_dtype),
                t5_ids.to(device=execution_device),
                t5xxl_weights=t5_weights.to(device=execution_device, dtype=torch.float32),
            )
        cond = cond.to(device=execution_device, dtype=model_dtype)
        if target_length is None:
            return cond
        return _anima_v3_pad_condition_to_length(cond, int(target_length))


@torch.no_grad()
def _anima_v3_encode_batch(
    pipe,
    prompts: List[str],
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
) -> torch.Tensor:
    if not enable_long_prompt or str(long_prompt_strategy).lower() == _ANIMA_LONG_PROMPT_FUSION_TRUNCATE:
        qwen_hidden, t5_ids, t5_weights = _anima_v3_prepare_condition_inputs(
            pipe,
            prompts,
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            t5_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        return _anima_v3_build_condition(
            pipe,
            qwen_hidden=qwen_hidden,
            t5_ids=t5_ids,
            t5_weights=t5_weights,
        )

    # Long-prompt mode is per prompt because each prompt can expand to a
    # different number of windows.  Concatenation mode can therefore produce
    # different sequence lengths per prompt, so pad the batch to the maximum.
    encoded = [
        _anima_v3_encode_long_single(
            pipe,
            p or "",
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )
        for p in prompts
    ]
    encoded = _anima_v3_pad_condition_list(encoded)
    return torch.cat(encoded, dim=0)


@torch.no_grad()
def _anima_v3_encode_single(
    pipe,
    text: str,
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
) -> torch.Tensor:
    return _anima_v3_encode_batch(
        pipe,
        [text or ""],
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
        enable_long_prompt=enable_long_prompt,
        long_prompt_strategy=long_prompt_strategy,
        long_prompt_chunk_size=long_prompt_chunk_size,
        long_prompt_chunk_decay=long_prompt_chunk_decay,
        long_prompt_strength=long_prompt_strength,
        long_prompt_anchor_tokens=long_prompt_anchor_tokens,
    )


@torch.no_grad()
def _anima_v3_mix_AND(
    pipe,
    text: str,
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    and_strength: float,
    base_bias: float,
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
) -> torch.Tensor:
    parts = _split_top_level_AND(text or "")
    parsed: List[Tuple[str, float]] = []
    for part in parts:
        t, w = _split_suffix_weight_top_level(part)
        t = (t or "").strip()
        if t:
            parsed.append((t, float(w)))
    if len(parsed) <= 1:
        return _anima_v3_encode_single(
            pipe,
            text or "",
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            enable_long_prompt=enable_long_prompt,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )

    clean_all, _spans, _has_weight = _anima_v3_weighted_spans(text or "")
    base = _anima_v3_encode_single(
        pipe,
        clean_all,
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
        enable_long_prompt=enable_long_prompt,
        long_prompt_strategy=long_prompt_strategy,
        long_prompt_chunk_size=long_prompt_chunk_size,
        long_prompt_chunk_decay=long_prompt_chunk_decay,
        long_prompt_strength=long_prompt_strength,
        long_prompt_anchor_tokens=long_prompt_anchor_tokens,
    )
    mixed = base * float(base_bias)
    denom = abs(float(base_bias)) + 1e-6
    for part_text, part_weight in parsed:
        cond_i = _anima_v3_encode_single(
            pipe,
            part_text,
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            enable_long_prompt=enable_long_prompt,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )
        target_length = max(int(base.shape[1]), int(mixed.shape[1]), int(cond_i.shape[1]))
        base = _anima_v3_pad_condition_to_length(base, target_length)
        mixed = _anima_v3_pad_condition_to_length(mixed, target_length)
        cond_i = _anima_v3_pad_condition_to_length(cond_i, target_length)
        wf = float(part_weight)
        if wf < 0.0:
            wf = 1.0 + wf
        wf = float(max(-2.0, min(3.0, wf)))
        mixed = mixed + cond_i * wf
        denom += abs(wf)
    mixed = mixed / denom
    base = _anima_v3_pad_condition_to_length(base, int(mixed.shape[1]))
    return base + (mixed - base) * float(and_strength)

def _anima_v3_expand_prompt_inputs(
    prompt: Union[str, List[str], Tuple[str, ...]],
    neg_prompt: Union[str, List[str], Tuple[str, ...], None],
    *,
    num_images_per_prompt: int,
) -> Tuple[List[str], List[str]]:
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = list(prompt)
    if len(prompts) == 0:
        raise ValueError("prompt must not be empty")

    if neg_prompt is None:
        negs = [""] * len(prompts)
    elif isinstance(neg_prompt, str):
        negs = [neg_prompt] * len(prompts)
    else:
        negs = list(neg_prompt)
        if len(negs) != len(prompts):
            raise ValueError(f"The number of prompts and neg_prompts not matched: {len(prompts)} / {len(negs)}")

    n = int(num_images_per_prompt)
    if n < 1:
        raise ValueError("num_images_per_prompt must be >= 1")
    expanded_prompts: List[str] = []
    expanded_negs: List[str] = []
    for p, ntext in zip(prompts, negs):
        for _ in range(n):
            expanded_prompts.append(p)
            expanded_negs.append(ntext)
    return expanded_prompts, expanded_negs


@torch.no_grad()
def get_weighted_text_embeddings_anima(
    pipe,
    prompt: Union[str, List[str]] = "",
    neg_prompt: Union[str, List[str], None] = "",
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: Optional[float] = 0.0,
    weight_clamp_max: Optional[float] = 3.0,
    enable_AND: bool = True,
    and_strength: float = 0.60,
    base_bias: float = 4.0,
    # Semantic prompt compilation options
    enable_semantic: bool = False,
    semantic_frontend: Optional[Any] = None,
    semantic_mode: str = "auto",
    semantic_target_t5_tokens: int = 480,
    semantic_qwen_input_max_tokens: int = 8192,
    semantic_compiler_max_new_tokens: int = 640,
    semantic_system_prompt: Optional[str] = None,
    semantic_tag_resolver: Optional[Any] = None,
    semantic_tag_resolver_path: Optional[Union[str, Path]] = None,
    semantic_process_negative: bool = False,
    semantic_allow_generation: bool = False,
    semantic_generation_kwargs: Optional[Dict[str, Any]] = None,
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Anima positive/negative conditioning with sd_embed prompt weights.

    This is compatible with the diffusers_anima pipeline layout where:
      - pipe.prompt_tokenizer contains qwen_tokenizer and t5_tokenizer
      - pipe.text_encoder encodes Qwen IDs into hidden states
      - pipe.transformer.preprocess_text_embeds builds the final (B,512,D) condition

    Weighting is applied in two places:
      1. Qwen hidden states are moved around a per-prompt mean anchor.
      2. Native T5 per-token weights are passed into preprocess_text_embeds.

    For unweighted prompts, the path is intentionally equivalent to
    pipe.encode_prompt(prompt, negative_prompt, num_images_per_prompt), except that
    batching and truncation are controlled here.
    """
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)
    try:
        prompt_list, neg_list = _anima_v3_expand_prompt_inputs(
            prompt,
            neg_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )

        if enable_AND and (any(_has_top_level_AND(p or "") for p in prompt_list) or any(_has_top_level_AND(n or "") for n in neg_list)):
            pos_list = [
                _anima_v3_mix_AND(
                    pipe,
                    p or "",
                    max_sequence_length=max_sequence_length,
                    qwen_weight_strength=qwen_weight_strength,
                    adapter_weight_strength=adapter_weight_strength,
                    weight_clamp_min=weight_clamp_min,
                    weight_clamp_max=weight_clamp_max,
                    and_strength=and_strength,
                    base_bias=base_bias,
                    enable_long_prompt=enable_long_prompt,
                    long_prompt_strategy=long_prompt_strategy,
                    long_prompt_chunk_size=long_prompt_chunk_size,
                    long_prompt_chunk_decay=long_prompt_chunk_decay,
                    long_prompt_strength=long_prompt_strength,
                    long_prompt_anchor_tokens=long_prompt_anchor_tokens,
                )
                for p in prompt_list
            ]
            neg_list_encoded = [
                _anima_v3_mix_AND(
                    pipe,
                    n or "",
                    max_sequence_length=max_sequence_length,
                    qwen_weight_strength=qwen_weight_strength,
                    adapter_weight_strength=adapter_weight_strength,
                    weight_clamp_min=weight_clamp_min,
                    weight_clamp_max=weight_clamp_max,
                    and_strength=and_strength,
                    base_bias=base_bias,
                    enable_long_prompt=enable_long_prompt,
                    long_prompt_strategy=long_prompt_strategy,
                    long_prompt_chunk_size=long_prompt_chunk_size,
                    long_prompt_chunk_decay=long_prompt_chunk_decay,
                    long_prompt_strength=long_prompt_strength,
                    long_prompt_anchor_tokens=long_prompt_anchor_tokens,
                )
                for n in neg_list
            ]
            target_len = max(
                max(int(x.shape[1]) for x in pos_list),
                max(int(x.shape[1]) for x in neg_list_encoded),
            )
            pos = torch.cat(_anima_v3_pad_condition_list(pos_list, target_len), dim=0)
            neg = torch.cat(_anima_v3_pad_condition_list(neg_list_encoded, target_len), dim=0)
            return pos, neg

        pos = _anima_v3_encode_batch(
            pipe,
            [p or "" for p in prompt_list],
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            enable_long_prompt=enable_long_prompt,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )
        neg = _anima_v3_encode_batch(
            pipe,
            [n or "" for n in neg_list],
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            enable_long_prompt=enable_long_prompt,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )
        return pos, neg
    finally:
        dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)


# ============================================================
# Anima Artist Mixer integration for get_weighted_text_embeddings_anima
# ============================================================
# This late override keeps all original sd_embed / diffusers_anima weighted
# encoding helpers above, and adds an optional transformer-side Artist Mixer
# patch for prompts such as:
#   {@vlizz:[style:1.0, pose:0.4], @ashraely:[style:0.6, body:0.8]}
#   @vlizz:[style:0.7, face:0.4, pose:1.2]
#
# Usage:
#   pos, neg = get_weighted_text_embeddings_anima(
#       pipe,
#       prompt="{@a:[style:1.0], @b:[eyes:0.6]}, 1girl",
#       enable_artist_mixer=True,
#   )
#   image = pipe(prompt_embeds=pos, negative_prompt_embeds=neg).images[0]
#   uninstall_anima_artist_mixer(pipe)

try:
    from sd_embed.anima_artist_mixer_plus_diffusers import (
        DiffusersAnimaArtistMixer as _SDEmbedDiffusersAnimaArtistMixer,
        parse_artist_mixer_syntax as _sd_embed_parse_artist_mixer_syntax,
        FUSION_INTERPOLATE as _SD_EMBED_MIXER_FUSION_INTERPOLATE,
        FUSION_BASE_PRESERVE as _SD_EMBED_MIXER_FUSION_BASE_PRESERVE,
        COMBINE_OUTPUT_AVG as _SD_EMBED_MIXER_COMBINE_OUTPUT_AVG,
        COMBINE_LOWRANK_AVG as _SD_EMBED_MIXER_COMBINE_LOWRANK_AVG,
    )
except Exception:  # The encoder remains usable even if the mixer file is not installed.
    _SDEmbedDiffusersAnimaArtistMixer = None
    _sd_embed_parse_artist_mixer_syntax = None
    _SD_EMBED_MIXER_FUSION_INTERPOLATE = "interpolate"
    _SD_EMBED_MIXER_FUSION_BASE_PRESERVE = "base_preserve"
    _SD_EMBED_MIXER_COMBINE_OUTPUT_AVG = "output_avg"
    _SD_EMBED_MIXER_COMBINE_LOWRANK_AVG = "lowrank_avg"


def _anima_artist_mixer_split_prompt_segments(text: str) -> List[str]:
    """Split a prompt by top-level commas while preserving bracketed mixer blocks."""
    s = str(text or "")
    out: List[str] = []
    buf: List[str] = []
    stack: List[str] = []
    pairs = {"[": "]", "{": "}", "(": ")"}
    quote = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            buf.append(ch); buf.append(s[i + 1]); i += 2; continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
        elif ch in pairs:
            stack.append(pairs[ch])
            buf.append(ch)
        elif stack and ch == stack[-1]:
            stack.pop()
            buf.append(ch)
        elif ch == "," and not stack:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _anima_artist_mixer_is_candidate(segment: str) -> bool:
    s = str(segment or "").strip()
    if not s or "@" not in s:
        return False
    if s.startswith("{") and s.endswith("}"):
        return ":" in s
    if s.startswith("::@") or s.startswith("@"):
        # Do not steal plain @artist tags from the normal Anima prompt.
        return (": [" in s) or (":" in s and "[" in s) or (s.count(":") >= 2) or ("::" in s)
    return False


def _anima_artist_mixer_parse_nonempty(syntax: str) -> bool:
    if _sd_embed_parse_artist_mixer_syntax is None:
        return False
    try:
        specs = _sd_embed_parse_artist_mixer_syntax(syntax or "")
        return bool(specs)
    except Exception:
        return False


def _anima_artist_mixer_extract_from_prompt(text: str) -> Tuple[str, List[str]]:
    """Return (clean_prompt, mixer_syntaxes) for top-level Artist Mixer syntax.

    The extractor is intentionally conservative: normal @artist tags are left in
    the prompt unless they use a component syntax or a grouped mixer syntax.
    """
    segments = _anima_artist_mixer_split_prompt_segments(text or "")
    clean: List[str] = []
    found: List[str] = []
    for seg in segments:
        raw = seg.strip()
        if _anima_artist_mixer_is_candidate(raw) and _anima_artist_mixer_parse_nonempty(raw):
            found.append(raw)
        else:
            clean.append(seg.strip())
    return ", ".join([x for x in clean if x]), found


def _anima_artist_mixer_extract_batch(prompts: List[str]) -> Tuple[List[str], Optional[str]]:
    cleaned: List[str] = []
    syntaxes: List[str] = []
    for p in prompts:
        c, found = _anima_artist_mixer_extract_from_prompt(p or "")
        cleaned.append(c)
        syntaxes.extend(found)
    if not syntaxes:
        return cleaned, None
    # A single transformer patch is global for the whole denoising pass.  If a
    # batch contains multiple different mixer blocks, merge them in declaration
    # order. This is predictable and avoids silently choosing only the first one.
    if len(syntaxes) == 1:
        return cleaned, syntaxes[0]
    return cleaned, "{" + ", ".join(s.strip()[1:-1].strip() if s.strip().startswith("{") and s.strip().endswith("}") else s.strip() for s in syntaxes) + "}"


def _anima_artist_mixer_make_encoder(
    pipe,
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
):
    """Artist encoder used by the mixer, sharing this file's Anima v3 path."""
    def _encode_artist(_pipe, text: str) -> torch.Tensor:
        return _anima_v3_encode_single(
            pipe,
            text or "",
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
            enable_long_prompt=enable_long_prompt,
            long_prompt_strategy=long_prompt_strategy,
            long_prompt_chunk_size=long_prompt_chunk_size,
            long_prompt_chunk_decay=long_prompt_chunk_decay,
            long_prompt_strength=long_prompt_strength,
            long_prompt_anchor_tokens=long_prompt_anchor_tokens,
        )
    return _encode_artist


def uninstall_anima_artist_mixer(pipe) -> None:
    """Remove the transformer patch created by get_weighted_text_embeddings_anima."""
    mixer = getattr(pipe, "_sd_embed_anima_artist_mixer", None)
    if mixer is not None:
        try:
            mixer.uninstall()
        finally:
            try:
                delattr(pipe, "_sd_embed_anima_artist_mixer")
            except Exception:
                setattr(pipe, "_sd_embed_anima_artist_mixer", None)


def get_anima_artist_mixer(pipe):
    """Return the currently installed Artist Mixer patcher, if any."""
    return getattr(pipe, "_sd_embed_anima_artist_mixer", None)


def _install_anima_artist_mixer_for_encoder(
    pipe,
    *,
    mixer_syntax: str,
    base_prompt: str,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
    strength: float,
    normalize_weights: bool,
    fusion_mode: str,
    combine_mode: str,
    lowrank_k: int,
    start_block: int,
    end_block: int,
    autoclean_previous: bool,
):
    if _SDEmbedDiffusersAnimaArtistMixer is None:
        raise ImportError(
            "anima_artist_mixer_plus_diffusers.py is required for enable_artist_mixer=True. "
            "Place it next to embedding_funcs.py or install it on PYTHONPATH."
        )
    if autoclean_previous:
        uninstall_anima_artist_mixer(pipe)
    encoder = _anima_artist_mixer_make_encoder(
        pipe,
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
        enable_long_prompt=enable_long_prompt,
        long_prompt_strategy=long_prompt_strategy,
        long_prompt_chunk_size=long_prompt_chunk_size,
        long_prompt_chunk_decay=long_prompt_chunk_decay,
        long_prompt_strength=long_prompt_strength,
        long_prompt_anchor_tokens=long_prompt_anchor_tokens,
    )
    mixer = _SDEmbedDiffusersAnimaArtistMixer(pipe)
    mixer.install(
        syntax=mixer_syntax,
        base_prompt=base_prompt,
        encode_artist_fn=encoder,
        strength=float(strength),
        normalize_weights=bool(normalize_weights),
        fusion_mode=str(fusion_mode),
        combine_mode=str(combine_mode),
        lowrank_k=int(lowrank_k),
        start_block=int(start_block),
        end_block=int(end_block),
    )
    setattr(pipe, "_sd_embed_anima_artist_mixer", mixer)
    return mixer


@torch.no_grad()
def get_weighted_text_embeddings_anima(
    pipe,
    prompt: Union[str, List[str]] = "",
    neg_prompt: Union[str, List[str], None] = "",
    *,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    lora_scale: Optional[float] = None,
    qwen_weight_strength: float = 1.25,
    adapter_weight_strength: float = 1.75,
    weight_clamp_min: Optional[float] = 0.0,
    weight_clamp_max: Optional[float] = 3.0,
    enable_AND: bool = True,
    and_strength: float = 0.60,
    base_bias: float = 4.0,
    # Semantic prompt compilation options
    enable_semantic: bool = False,
    semantic_frontend: Optional[Any] = None,
    semantic_mode: str = "auto",
    semantic_target_t5_tokens: int = 480,
    semantic_qwen_input_max_tokens: int = 8192,
    semantic_compiler_max_new_tokens: int = 640,
    semantic_system_prompt: Optional[str] = None,
    semantic_tag_resolver: Optional[Any] = None,
    semantic_tag_resolver_path: Optional[Union[str, Path]] = None,
    semantic_process_negative: bool = False,
    semantic_allow_generation: bool = False,
    semantic_generation_kwargs: Optional[Dict[str, Any]] = None,
    # Long prompt options
    enable_long_prompt: bool = True,
    long_prompt_strategy: str = _ANIMA_LONG_PROMPT_FUSION_CHUNK_CONCAT,
    long_prompt_chunk_size: Optional[int] = None,
    long_prompt_chunk_decay: float = 1.0,
    long_prompt_strength: float = 1.0,
    long_prompt_anchor_tokens: int = 160,
    # Artist Mixer options
    enable_artist_mixer: bool = False,
    artist_mixer: Optional[str] = None,
    artist_mixer_strength: float = 1.0,
    artist_mixer_normalize_weights: bool = True,
    artist_mixer_fusion_mode: str = "interpolate",
    artist_mixer_combine_mode: str = "output_avg",
    artist_mixer_lowrank_k: int = 1,
    artist_mixer_start_block: int = 0,
    artist_mixer_end_block: int = -1,
    artist_mixer_autoclean_previous: bool = True,
    return_artist_mixer: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Any]]:
    """Return Anima positive/negative conditioning with optional Artist Mixer.

    `enable_semantic=True` integrates the inference-only Qwen Base semantic prompt
    compiler into the weighted Anima path. The semantic stage runs *before* the
    existing LPW/AND/long-prompt encoder so the final conditioning still comes
    from `get_weighted_text_embeddings_anima`, but top-level `AND`, `BREAK`,
    pure weighted items like `(red hair:1.4)`, and Artist Mixer syntax are
    preserved structurally.

    `enable_artist_mixer=True` extracts top-level Artist Mixer syntax from the
    positive prompt and installs a transformer patch that remains active for the
    following pipeline call.  Call `uninstall_anima_artist_mixer(pipe)` after
    generation, or pass `return_artist_mixer=True` and call `mixer.uninstall()`.

    You may also pass `artist_mixer="{@a:[style:1.0], @b:[eyes:0.6]}"` to keep
    mixer syntax outside of the visible prompt.
    """
    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)
    mixer_obj = None
    try:
        prompt_list, neg_list = _anima_v3_expand_prompt_inputs(
            prompt,
            neg_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )

        mixer_syntax = artist_mixer.strip() if isinstance(artist_mixer, str) and artist_mixer.strip() else None
        if enable_artist_mixer:
            cleaned_prompt_list, extracted = _anima_artist_mixer_extract_batch(prompt_list)
            prompt_list = cleaned_prompt_list
            if mixer_syntax is None:
                mixer_syntax = extracted
        elif mixer_syntax is not None:
            # Explicit mixer string means: install the patch, but do not force users
            # to put the mixer syntax in the visible prompt.
            enable_artist_mixer = True

        if enable_semantic:
            semantic_compiler = semantic_frontend
            if semantic_compiler is None:
                semantic_compiler = _anima_semantic_make_frontend(
                    pipe,
                    mode=semantic_mode,
                    target_t5_tokens=semantic_target_t5_tokens,
                    qwen_input_max_tokens=semantic_qwen_input_max_tokens,
                    compiler_max_new_tokens=semantic_compiler_max_new_tokens,
                    system_prompt=semantic_system_prompt,
                    tag_resolver=semantic_tag_resolver,
                    tag_resolver_path=semantic_tag_resolver_path,
                    process_negative_prompt=semantic_process_negative,
                    allow_generation=semantic_allow_generation,
                    generation_kwargs=semantic_generation_kwargs,
                )
            prompt_list = [
                _anima_semantic_compile_prompt(semantic_compiler, p or "", negative=False)
                for p in prompt_list
            ]
            neg_list = [
                _anima_semantic_compile_prompt(semantic_compiler, n or "", negative=True)
                for n in neg_list
            ]

        if enable_artist_mixer and mixer_syntax:
            base_for_artist = prompt_list[0] if prompt_list else ""
            mixer_obj = _install_anima_artist_mixer_for_encoder(
                pipe,
                mixer_syntax=mixer_syntax,
                base_prompt=base_for_artist,
                max_sequence_length=max_sequence_length,
                qwen_weight_strength=qwen_weight_strength,
                adapter_weight_strength=adapter_weight_strength,
                weight_clamp_min=weight_clamp_min,
                weight_clamp_max=weight_clamp_max,
                enable_long_prompt=enable_long_prompt,
                long_prompt_strategy=long_prompt_strategy,
                long_prompt_chunk_size=long_prompt_chunk_size,
                long_prompt_chunk_decay=long_prompt_chunk_decay,
                long_prompt_strength=long_prompt_strength,
                long_prompt_anchor_tokens=long_prompt_anchor_tokens,
                strength=artist_mixer_strength,
                normalize_weights=artist_mixer_normalize_weights,
                fusion_mode=artist_mixer_fusion_mode,
                combine_mode=artist_mixer_combine_mode,
                lowrank_k=artist_mixer_lowrank_k,
                start_block=artist_mixer_start_block,
                end_block=artist_mixer_end_block,
                autoclean_previous=artist_mixer_autoclean_previous,
            )

        if enable_AND and (any(_has_top_level_AND(p or "") for p in prompt_list) or any(_has_top_level_AND(n or "") for n in neg_list)):
            pos_list = [
                _anima_v3_mix_AND(
                    pipe,
                    p or "",
                    max_sequence_length=max_sequence_length,
                    qwen_weight_strength=qwen_weight_strength,
                    adapter_weight_strength=adapter_weight_strength,
                    weight_clamp_min=weight_clamp_min,
                    weight_clamp_max=weight_clamp_max,
                    and_strength=and_strength,
                    base_bias=base_bias,
                    enable_long_prompt=enable_long_prompt,
                    long_prompt_strategy=long_prompt_strategy,
                    long_prompt_chunk_size=long_prompt_chunk_size,
                    long_prompt_chunk_decay=long_prompt_chunk_decay,
                    long_prompt_strength=long_prompt_strength,
                    long_prompt_anchor_tokens=long_prompt_anchor_tokens,
                )
                for p in prompt_list
            ]
            neg_list_encoded = [
                _anima_v3_mix_AND(
                    pipe,
                    n or "",
                    max_sequence_length=max_sequence_length,
                    qwen_weight_strength=qwen_weight_strength,
                    adapter_weight_strength=adapter_weight_strength,
                    weight_clamp_min=weight_clamp_min,
                    weight_clamp_max=weight_clamp_max,
                    and_strength=and_strength,
                    base_bias=base_bias,
                    enable_long_prompt=enable_long_prompt,
                    long_prompt_strategy=long_prompt_strategy,
                    long_prompt_chunk_size=long_prompt_chunk_size,
                    long_prompt_chunk_decay=long_prompt_chunk_decay,
                    long_prompt_strength=long_prompt_strength,
                    long_prompt_anchor_tokens=long_prompt_anchor_tokens,
                )
                for n in neg_list
            ]
            target_len = max(
                max(int(x.shape[1]) for x in pos_list),
                max(int(x.shape[1]) for x in neg_list_encoded),
            )
            pos = torch.cat(_anima_v3_pad_condition_list(pos_list, target_len), dim=0)
            neg = torch.cat(_anima_v3_pad_condition_list(neg_list_encoded, target_len), dim=0)
        else:
            pos = _anima_v3_encode_batch(
                pipe,
                [p or "" for p in prompt_list],
                max_sequence_length=max_sequence_length,
                qwen_weight_strength=qwen_weight_strength,
                adapter_weight_strength=adapter_weight_strength,
                weight_clamp_min=weight_clamp_min,
                weight_clamp_max=weight_clamp_max,
                enable_long_prompt=enable_long_prompt,
                long_prompt_strategy=long_prompt_strategy,
                long_prompt_chunk_size=long_prompt_chunk_size,
                long_prompt_chunk_decay=long_prompt_chunk_decay,
                long_prompt_strength=long_prompt_strength,
                long_prompt_anchor_tokens=long_prompt_anchor_tokens,
            )
            neg = _anima_v3_encode_batch(
                pipe,
                [n or "" for n in neg_list],
                max_sequence_length=max_sequence_length,
                qwen_weight_strength=qwen_weight_strength,
                adapter_weight_strength=adapter_weight_strength,
                weight_clamp_min=weight_clamp_min,
                weight_clamp_max=weight_clamp_max,
                enable_long_prompt=enable_long_prompt,
                long_prompt_strategy=long_prompt_strategy,
                long_prompt_chunk_size=long_prompt_chunk_size,
                long_prompt_chunk_decay=long_prompt_chunk_decay,
                long_prompt_strength=long_prompt_strength,
                long_prompt_anchor_tokens=long_prompt_anchor_tokens,
            )

        pos, neg = _anima_v3_align_pos_neg_conditions(pos, neg)
        if return_artist_mixer:
            return pos, neg, mixer_obj
        return pos, neg
    finally:
        dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)


# ============================================================
# Anima long-prompt BREAK-aware token packing override
# ============================================================

def _anima_v3_split_top_level_comma_items(text: str) -> List[str]:
    text = text or ""
    if not text:
        return [""]
    items: List[str] = []
    buf: List[str] = []
    par = brk = brace = 0
    escape = False
    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == "(":
            par += 1
        elif ch == ")" and par > 0:
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]" and brk > 0:
            brk -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1
        if ch == "," and par == 0 and brk == 0 and brace == 0:
            items.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    items.append("".join(buf).strip())
    return items


def _anima_v3_count_prompt_tokens_dual(qwen_tokenizer, t5_tokenizer, text: str, *, qwen_pad: int, t5_eos: int) -> Tuple[int, int]:
    q_ids, _q_weights, _clean_q, _has_q = _anima_v3_tokenize_with_weights(
        qwen_tokenizer,
        text or "",
        empty_token_id=int(qwen_pad),
        add_eos=False,
        eos_token_id=None,
        truncate_to=None,
    )
    t_ids, _t_weights, _clean_t, _has_t = _anima_v3_tokenize_with_weights(
        t5_tokenizer,
        text or "",
        empty_token_id=int(t5_eos),
        add_eos=True,
        eos_token_id=int(t5_eos),
        truncate_to=None,
    )
    return len(q_ids), len(t_ids)


def _anima_v3_pack_prompt_text_with_breaks(
    qwen_tokenizer,
    t5_tokenizer,
    text: str,
    *,
    chunk_size: int,
    break_token: str = "BREAK",
) -> List[str]:
    qwen_pad = getattr(qwen_tokenizer, "pad_token_id", None)
    if qwen_pad is None:
        qwen_pad = _ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_eos = getattr(t5_tokenizer, "eos_token_id", None)
    if t5_eos is None:
        t5_eos = 1

    items = _anima_v3_split_top_level_comma_items(text or "")
    out_chunks: List[str] = []
    cur_items: List[str] = []

    def join_items(parts: List[str]) -> str:
        return ", ".join([p for p in parts if p is not None and str(p).strip() != ""])

    def flush_current() -> None:
        nonlocal cur_items
        chunk_text = join_items(cur_items)
        if chunk_text.strip():
            out_chunks.append(chunk_text)
        cur_items = []

    def fits_text(candidate_text: str) -> bool:
        q_len, t_len = _anima_v3_count_prompt_tokens_dual(
            qwen_tokenizer,
            t5_tokenizer,
            candidate_text,
            qwen_pad=int(qwen_pad),
            t5_eos=int(t5_eos),
        )
        return q_len <= int(chunk_size) and t_len <= int(chunk_size)

    for item in items:
        stripped = (item or "").strip()
        if not stripped:
            continue
        if stripped.upper() == str(break_token).upper():
            flush_current()
            continue

        # Oversized single item: fall back to native weighted text splitter using
        # actual tokenizer budgets. BREAK still acts as a hard boundary around it.
        if not fits_text(stripped):
            flush_current()
            subchunks = _anima_v3_split_prompt_text_dual_budget(
                qwen_tokenizer,
                t5_tokenizer,
                stripped,
                chunk_size=int(chunk_size),
            )
            for sub in subchunks:
                sub = (sub or "").strip()
                if sub:
                    out_chunks.append(sub)
            continue

        candidate_items = [*cur_items, stripped]
        candidate_text = join_items(candidate_items)
        if not cur_items or fits_text(candidate_text):
            cur_items = candidate_items
            continue

        flush_current()
        cur_items = [stripped]

    flush_current()
    return out_chunks or [text or ""]


@torch.no_grad()
def _anima_v3_encode_long_single(
    pipe,
    text: str,
    *,
    max_sequence_length: int,
    qwen_weight_strength: float,
    adapter_weight_strength: float,
    weight_clamp_min: Optional[float],
    weight_clamp_max: Optional[float],
    long_prompt_strategy: str,
    long_prompt_chunk_size: Optional[int],
    long_prompt_chunk_decay: float,
    long_prompt_strength: float,
    long_prompt_anchor_tokens: int,
) -> torch.Tensor:
    _prompt_tokenizer, qwen_tokenizer, t5_tokenizer = _anima_v3_get_prompt_tokenizer(pipe)
    qwen_pad = getattr(qwen_tokenizer, "pad_token_id", None)
    if qwen_pad is None:
        qwen_pad = _ANIMA_QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_eos = getattr(t5_tokenizer, "eos_token_id", None)
    if t5_eos is None:
        t5_eos = 1

    q_ids, _q_weights, _clean_q, _has_q = _anima_v3_tokenize_with_weights(
        qwen_tokenizer,
        text or "",
        empty_token_id=int(qwen_pad),
        add_eos=False,
        eos_token_id=None,
        truncate_to=None,
    )
    t_ids, _t_weights, _clean_t, _has_t = _anima_v3_tokenize_with_weights(
        t5_tokenizer,
        text or "",
        empty_token_id=int(t5_eos),
        add_eos=True,
        eos_token_id=int(t5_eos),
        truncate_to=None,
    )

    chunk_size = _anima_v3_long_prompt_chunk_size(max_sequence_length, long_prompt_chunk_size)

    if len(q_ids) <= chunk_size and len(t_ids) <= chunk_size:
        qwen_hidden, t5_ids, t5_weights = _anima_v3_prepare_condition_inputs(
            pipe,
            [text or ""],
            max_sequence_length=chunk_size,
            qwen_weight_strength=qwen_weight_strength,
            t5_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        return _anima_v3_build_condition(
            pipe,
            qwen_hidden=qwen_hidden,
            t5_ids=t5_ids,
            t5_weights=t5_weights,
        )

    text_chunks = _anima_v3_pack_prompt_text_with_breaks(
        qwen_tokenizer,
        t5_tokenizer,
        text or "",
        chunk_size=chunk_size,
        break_token="BREAK",
    )

    chunk_conditions: List[torch.Tensor] = []
    for chunk_text in text_chunks:
        qwen_hidden, t5_ids_t, t5_weights_t = _anima_v3_prepare_condition_inputs(
            pipe,
            [chunk_text or ""],
            max_sequence_length=chunk_size,
            qwen_weight_strength=qwen_weight_strength,
            t5_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        chunk_conditions.append(_anima_v3_build_condition(
            pipe,
            qwen_hidden=qwen_hidden,
            t5_ids=t5_ids_t,
            t5_weights=t5_weights_t,
            target_length=_ANIMA_CONDITIONING_MAX_LENGTH,
        ))

    return _anima_v3_fuse_long_prompt_conditions(
        chunk_conditions,
        strength=long_prompt_strength,
        chunk_decay=long_prompt_chunk_decay,
        mode=long_prompt_strategy,
        anchor_tokens=long_prompt_anchor_tokens,
    )
