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

def _get_zimage_prompt_tokens_and_weights(
    tokenizer,
    prompt: str,
) -> Tuple[List[int], List[float]]:
    if prompt is None or len(prompt) == 0:
        return [], []

    texts_and_weights = parse_prompt_attention(prompt)
    token_ids: List[int] = []
    weights: List[float] = []

    for word, weight in texts_and_weights:
        if not word:
            continue

        enc = tokenizer(
            word,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        ids = enc.input_ids[0].tolist()
        if not ids:
            continue

        token_ids.extend(ids)
        weights.extend([weight] * len(ids))

    return token_ids, weights


def get_weighted_text_embeddings_zimage(
    pipe: DiffusionPipeline,
    prompt: Union[str, List[str]]        = "",
    neg_prompt: Union[str, List[str]]    = "",
    max_sequence_length: int             = 1024,
    lora_scale: Optional[float]          = None,
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
            raise ValueError(
                f"The number of prompts and neg_prompts not matched: {len(prompt_list)} / {len(neg_list)}"
            )

    device = pipe.device
    tokenizer = pipe.tokenizer

    dynamically_scale_lora_layers(pipe, lora_scale=lora_scale)

    def _encode_single(text: str) -> torch.Tensor:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            templated = text

        text_inputs = tokenizer(
            templated,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)            # (1, max_seq_len)
        attn_mask = text_inputs.attention_mask.to(device).bool()  # (1, max_seq_len)

        with torch.no_grad():
            outputs = pipe.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states[-2]: (1, max_seq_len, H)
        hidden_all = outputs.hidden_states[-2][0]      # (max_seq_len, H)
        valid_mask = attn_mask[0]                      # (max_seq_len,)
        full_ids = input_ids[0][valid_mask]            # (L, )
        hidden = hidden_all[valid_mask]                # (L, H)

        content_ids, content_weights = _get_zimage_prompt_tokens_and_weights(
            tokenizer, text
        )

        w_full = [1.0] * full_ids.numel()

        if content_ids:
            full_ids_list = full_ids.tolist()
            cid_list = content_ids

            start_idx = -1
            max_start = len(full_ids_list) - len(cid_list)
            for s in range(max_start + 1):
                if full_ids_list[s:s + len(cid_list)] == cid_list:
                    start_idx = s
                    break

            if start_idx != -1:
                for j, wt in enumerate(content_weights):
                    pos = start_idx + j
                    if pos >= len(w_full):
                        break
                    w_full[pos] = wt

        weight_tensor = torch.tensor(
            w_full,
            dtype=hidden.dtype,
            device=hidden.device,
        )

        hidden = _apply_weights_vec_scale(hidden, weight_tensor)
        return hidden

    prompt_embeds: List[torch.Tensor] = []
    neg_embeds: List[torch.Tensor] = []

    for p in prompt_list:
        prompt_embeds.append(_encode_single(p))

    for n in neg_list:
        neg_embeds.append(_encode_single(n))

    dynamically_unscale_lora_layers(pipe, lora_scale=lora_scale)

    return prompt_embeds, neg_embeds