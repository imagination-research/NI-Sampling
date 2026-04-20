import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierBlock(nn.Module):
    '''
    This implementation is heavily based on https://github.com/thu-nics/R2R/blob/main/r2r/models/router.py
    '''
    
    def __init__(self, in_dim, out_dim, expansion_factor, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.layer_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * expansion_factor, out_dim),
            nn.Dropout(dropout),
        )
        
        self.shortcut = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )
        
    def forward(self, x):
        normalized = self.layer_norm(x)
        residual = self.shortcut(x)
        return residual + self.mlp(normalized)
        

class Indicator(nn.Module):
    
    def __init__(self, depth, width, token_emb_dim, hidden_states_dim, topk=None, topk_norm=False, concat_proj=False,
                 use_positional_embedding=False,
                 input_topk_token=1,
                 dropout=0.2):
        super().__init__()
        self.depth = depth
        
        self.token_emb_proj = nn.Linear(token_emb_dim, width)
        self.hidden_states_proj = nn.Linear(hidden_states_dim, width)
        concat_factor = 2
        self.topk = topk
        self.topk_norm = topk_norm
        if topk is not None and topk != 0:
            self.logits_proj = nn.Linear(topk, width)
            concat_factor += 1
        else:
            self.logits_proj = None
            
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            from positional_embedding import LlamaRotaryEmbedding
            self.pos_embedding = LlamaRotaryEmbedding(dim=width)
            concat_factor += 2
        
        self.input_topk_token = input_topk_token
        if input_topk_token > 1:
            self.extra_token_proj = nn.Linear((input_topk_token-1)*token_emb_dim, width)
            concat_factor += 1
        else:
            self.extra_token_proj = None
        
        in_dim = concat_factor * width
        self.concat_proj_enable = concat_proj
        if concat_proj:
            self.concat_proj = nn.Linear(in_dim, in_dim)
        
        self.backbone = nn.ModuleList()
        for i in range(depth):
            self.backbone.append(ClassifierBlock(in_dim, width, expansion_factor=4, dropout=dropout))
            in_dim = width
            
        self.out_layer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2),
        )
    
    def forward(self, hidden_states, token_emb, logits=None, position_ids=None, extra_token_emb=None):
        """
        hidden_states: [B, L, C]
        token_emb: [B, L, C]
        token_logits: [B, L, topk]
        position_ids: [B, L], currently not supported (because LLaDA model doesn't have positional emb)
        extra_token_emb: [B, L, l, C]
        """
        hidden_states = self.hidden_states_proj(hidden_states)
        token_emb = self.token_emb_proj(token_emb)
        if self.logits_proj is not None:
            logits_emb = self.logits_proj(logits)
            x = torch.cat([hidden_states, token_emb, logits_emb], dim=-1)
        else:
            x = torch.cat([hidden_states, token_emb], dim=-1)
            
        # positional embedding
        if self.use_positional_embedding:
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)
            pos_embedding = self.pos_embedding(x, pos)
            x = torch.cat([x, pos_embedding[0], pos_embedding[1]], dim=-1)
            
        # extra token
        if self.extra_token_proj is not None:
            assert extra_token_emb is not None
            B, L, l, C = extra_token_emb.shape
            extra_token_emb = extra_token_emb.reshape(B, L, l*C)
            extra_token_emb = self.extra_token_proj(extra_token_emb)
            x = torch.cat([x, extra_token_emb], dim=-1)

            
        if self.concat_proj_enable:
            x = self.concat_proj(x)
            
        for i in range(self.depth):
            x = self.backbone[i](x)
            
        output = self.out_layer(x)
        
        return output
        
        
        