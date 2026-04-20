import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import copy
import yaml
import time
import random
from ni_sampling.indicator import *


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # import ipdb; ipdb.set_trace()
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log((- torch.log(noise + 1e-10)) + 1e-10)
    return logits + temperature * gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def ni_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             prob_threshold=None,
             indicator=None, indicator_threshold=None,
             ):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    assert indicator is not None, "NI Sampling needs a trained indicator"

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            nfe += 1
            if cfg_scale > 0.: # unused
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                outputs = model(x, output_hidden_states=True)
                logits = outputs.logits
                hidden_states_input = outputs.hidden_states[-1][mask_index][None].to(torch.float32)
                if getattr(indicator, "use_sampled_token", False):
                    sampled_indices = (x != mask_id)
                    logits[sampled_indices] = torch.zeros_like(logits[sampled_indices]) - indicator.sampled_token_logits
                    temp_tensor = logits[sampled_indices]
                    temp_tensor[torch.arange(sampled_indices.nonzero().shape[0]), x[sampled_indices]] = \
                        indicator.sampled_token_logits
                    logits[sampled_indices] = temp_tensor
                            
                input_topk_token = getattr(indicator, "input_topk_token", None)
                                
                logits_sample = add_gumbel_noise(logits, temperature=temperature)
                sampled_token = logits_sample.argmax(dim=-1)[mask_index][None]
                if input_topk_token is not None and input_topk_token != 1:
                    extra_token = torch.topk(logits, k=input_topk_token, dim=-1).indices[:, :, 1:][mask_index][None]
                sampled_token_emb_input = model.model.transformer.wte(sampled_token).to(torch.float32)
                if input_topk_token is not None and input_topk_token != 1:
                    extra_token_emb = model.model.transformer.wte(extra_token).to(torch.float32)
                else:
                    extra_token_emb = None
                if indicator.topk is not None and indicator.topk != 0:
                    leng = logits[mask_index].shape[0]
                    if indicator.topk_norm:
                        topk_logits = torch.topk(logits[mask_index][None].softmax(dim=-1), k=indicator.topk, dim=-1)[0].to(torch.float32)
                        sampled_logits = logits[mask_index].softmax(dim=-1)[torch.arange(leng), sampled_token[0]]
                    else:
                        topk_logits = torch.topk(logits[mask_index][None], k=indicator.topk, dim=-1)[0].to(torch.float32)
                        sampled_logits = logits[mask_index].softmax(dim=-1)[torch.arange(leng), sampled_token[0]]
                    if temperature > 0:
                        topk_logits[0][:,0] = sampled_logits
                    
            if indicator.logits_proj is not None:
                pred_labels = indicator(hidden_states_input, sampled_token_emb_input, topk_logits, extra_token_emb=extra_token_emb)
            else:
                pred_labels = indicator(hidden_states_input, sampled_token_emb_input, extra_token_emb=extra_token_emb)
            pred_labels = pred_labels.softmax(dim=-1)
            # same as original sampling, to ensure that the token with highest confidence is selected
            # -------------------------------------------
            # logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            logits_with_noise = logits_sample
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            x0_greedy = logits.argmax(dim=-1)
            x0_greedy = torch.where(mask_index, x0_greedy, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index_indicator = torch.zeros_like(x0_greedy, dtype=torch.bool, device=x0.device)
            
            # -------------------------------------------
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if prob_threshold is not None:
                    transfer_index[j][confidence[j]>=prob_threshold] = True
                traj_preserve_indices = (pred_labels[j, :, 1] > indicator_threshold).nonzero()[:, 0]
                traj_pred_indices = mask_index.nonzero()[:, 1][traj_preserve_indices]
                transfer_index_indicator[j][traj_pred_indices] = True
            x[transfer_index] = x0[transfer_index]
            x[transfer_index_indicator] = x0_greedy[transfer_index_indicator]
            
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        
    return x, nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        try:
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        except:
            import ipdb; ipdb.set_trace()
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    max_pos = confidence[0].argmax().item()
    return x0, transfer_index, max_pos
