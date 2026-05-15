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
import typing
import traceback

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

        empty_qwen_out = text_encoder(
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

    qwen_outputs = text_encoder(
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
            out = text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask)
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


@torch.no_grad()
def _anima_v3_build_condition(
    pipe,
    *,
    qwen_hidden: torch.Tensor,
    t5_ids: torch.Tensor,
    t5_weights: torch.Tensor,
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
        if cond.shape[1] < _ANIMA_CONDITIONING_MAX_LENGTH:
            cond = torch.nn.functional.pad(cond, (0, 0, 0, _ANIMA_CONDITIONING_MAX_LENGTH - cond.shape[1]))
        elif cond.shape[1] > _ANIMA_CONDITIONING_MAX_LENGTH:
            cond = cond[:, :_ANIMA_CONDITIONING_MAX_LENGTH, :]
        return cond


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
) -> torch.Tensor:
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
) -> torch.Tensor:
    return _anima_v3_encode_batch(
        pipe,
        [text or ""],
        max_sequence_length=max_sequence_length,
        qwen_weight_strength=qwen_weight_strength,
        adapter_weight_strength=adapter_weight_strength,
        weight_clamp_min=weight_clamp_min,
        weight_clamp_max=weight_clamp_max,
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
        )
        wf = float(part_weight)
        if wf < 0.0:
            wf = 1.0 + wf
        wf = float(max(-2.0, min(3.0, wf)))
        mixed = mixed + cond_i * wf
        denom += abs(wf)
    mixed = mixed / denom
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
            pos = torch.cat([
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
                )
                for p in prompt_list
            ], dim=0)
            neg = torch.cat([
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
                )
                for n in neg_list
            ], dim=0)
            return pos, neg

        pos = _anima_v3_encode_batch(
            pipe,
            [p or "" for p in prompt_list],
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        neg = _anima_v3_encode_batch(
            pipe,
            [n or "" for n in neg_list],
            max_sequence_length=max_sequence_length,
            qwen_weight_strength=qwen_weight_strength,
            adapter_weight_strength=adapter_weight_strength,
            weight_clamp_min=weight_clamp_min,
            weight_clamp_max=weight_clamp_max,
        )
        return pos, neg
    finally:
        dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)
