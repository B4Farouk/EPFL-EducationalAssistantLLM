from transformers import (
    AutoTokenizer, RobertaModel, AutoConfig, 
    RobertaPreTrainedModel, RobertaConfig
)

import torch
import torch.nn as nn
import torch.nn.init as init

from typing import List


########################################
# REWARD HEADS
########################################

class TransformerHead(nn.Module):
    """Transformer head for the reward model
    """
    
    class PositionalEncoding(nn.Module):
        """Positional encoding for the transformer head
        """
        
        def __init__(self, hiddens_num, hiddens_dim, dropout=0.1):
            super().__init__()
                
            numerator = torch.arange(0, hiddens_num, dtype=torch.float32).reshape(-1, 1)
            denominator = torch.pow(10_000, torch.arange(0, hiddens_dim, 2, dtype=torch.float32) / hiddens_dim)
            p = numerator / denominator
        
            self.positional_encoding = torch.zeros((1, hiddens_num, hiddens_dim), requires_grad=False)
            self.positional_encoding[:, :, 0::2] = torch.sin(p)
            self.positional_encoding[:, :, 1::2] = torch.cos(p)
        
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            pos_enc = self.dropout(x + self.positional_encoding.to(x.device))
            return pos_enc
   
    
    def __init__(self, input_dim: int, max_seq_len: int, nheads :int, 
                dropout: float =0.1):
        super().__init__().__init__()
            
        # positional encoding        
        self.positional_encoding = TransformerHead.PositionalEncoding(max_seq_len, input_dim)

        # altered transformer encoder
        self.encoder_attn = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=nheads, dropout=dropout,
                batch_first=True)
        self.encoder_attn_dropout = nn.Dropout(dropout)
        self.encoder_norm = nn.LayerNorm(input_dim)
        
        self.encoder_ff = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(2 * input_dim, input_dim // 32),
            nn.Dropout(dropout)
        )
        self.skip_encoder_ff = nn.Sequential(
            nn.Linear(input_dim, input_dim // 32),
            nn.Dropout(dropout)
        )
        self.encoder_ff_norm = nn.LayerNorm(input_dim // 32)
        
        # projection
        self.projection = nn.Sequential(
            nn.Linear(max_seq_len * input_dim // 32, max_seq_len * input_dim // (32 * 24)),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(max_seq_len * input_dim // (32 * 24), max_seq_len * input_dim // (32 * 24 * 8)),
            nn.Dropout(dropout)
        )
        self.skip_projection = nn.Sequential(
            nn.Linear(max_seq_len * input_dim // 32, max_seq_len * input_dim // (32 * 24 * 8)),
            nn.Dropout(dropout)
        )
        self.projection_norm = nn.LayerNorm(max_seq_len * input_dim // (32 * 24 * 8))
        
        # interpolation
        self.interpolation = nn.Sequential(
            nn.Linear(max_seq_len * input_dim // (32 * 24 * 8), 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, inputs, padding_mask=None):
        # transformer: self-attention layer
        outputs = self.positional_encoding(inputs)
        attn_outputs = self.encoder_attn(key=outputs, query=outputs, value=outputs, key_padding_mask=padding_mask, need_weights=False)[0]        
        outputs = self.encoder_norm(outputs + self.encoder_attn_dropout(attn_outputs))
            
        # transformer: feed forward
        outputs = self.encoder_ff_norm(self.encoder_ff(outputs) + self.skip_encoder_ff(outputs))
        
        # projection
        outputs = outputs.reshape(inputs.shape[0], -1)
        outputs = self.projection_norm(self.projection(outputs) + self.skip_projection(outputs))
        
        # interpolation
        reward = self.interpolation(outputs)
        return reward
     
########################################
# REWARD MODEL
########################################

class RewardModelConfig(RobertaConfig):
    """Reward model configuration class
    """
    model_type = "RewardModel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # reward model
        self.max_seq_len = 512
        self.problem_type = "regression"
        
        # base
        self.base_frozen = True
        
        # head
        self.head_dropout = 0.1
        self.head_tsf_nheads = 1
        self.head_tsf_mask_padding = True

class RewardModel(RobertaPreTrainedModel):
    """Reward Model class
    """
    config_class  = RewardModelConfig
    
    def __init__(self, config =RewardModelConfig()):
        super().__init__(config)
        
        # base for the reward model
        base_id = "roberta-base" # HF pretrained model ID
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        self.base = RobertaModel(AutoConfig.from_pretrained(base_id))
        # freeze the base
        if self.config.base_frozen:
            self.base = self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False
 
        # regression head to compute the reward
        self.head = TransformerHead(
            input_dim=self.base.config.hidden_size,
            max_seq_len=self.config.max_seq_len,
            nheads=self.config.head_tsf_nheads,
            dropout=self.config.head_dropout
        )
        
        # initialize the parameters
        self.__reset_head_parameters()
       
    def tokenize(self, texts: List[str]):
        with torch.no_grad():
            encoding = self.tokenizer(texts,
                max_length=self.config.max_seq_len, truncation=True, padding="max_length",
                return_tensors="pt", return_attention_mask=True)
            
            # create padding tokens mask from head's self-attention
            if self.config.head_tsf_mask_padding:
                encoding["head_padding_mask"] = (encoding["input_ids"] == self.tokenizer.pad_token_id).bool()
            
            return encoding
    
    def forward(self, input_ids, attention_mask, head_padding_mask=None):
        base_encodings = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        reward = self.head(inputs=base_encodings, padding_mask=head_padding_mask)
        return reward

    def get_rewards(self, demonstrations):
        with torch.no_grad():
            rewards = []
            for pair in demonstrations:
                # compute the reward for each pair
                encoded = self.tokenize([pair["chosen"], pair["rejected"]])
                scores = self.forward(**encoded).squeeze()
                # append the rewards
                rewards.append({
                    'chosen': scores[0].item(),
                    'rejected': scores[1].item()
                })
            
            return rewards
    
    def parameters(self):
        ### to ensure that only the head is trained
        return self.head.parameters()
        
    def named_parameters(self):
        return self.head.named_parameters()
    
    def train(self, mode: bool =True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        
        self.training = mode
        self.head.train(mode)
        
        return self

    def __reset_head_parameters(self):
        # initialize the head parameters
        for param in self.head.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)
        
        # copy the self-attention parameters of the last layer of the base to the head's self-attention
        bert_last_layer_attn_in_weights_t = torch.cat([
            self.base.encoder.layer[-1].attention.self.query.weight.data,
            self.base.encoder.layer[-1].attention.self.key.weight.data,
            self.base.encoder.layer[-1].attention.self.value.weight.data
        ], dim=0).data
        bert_last_layer_attn_in_bias_t = torch.cat([
            self.base.encoder.layer[-1].attention.self.query.bias.data,
            self.base.encoder.layer[-1].attention.self.key.bias.data,
            self.base.encoder.layer[-1].attention.self.value.bias.data
        ], dim=0).data
        self.head.encoder_attn.in_proj_weight.data.copy_(bert_last_layer_attn_in_weights_t)
        self.head.encoder_attn.in_proj_bias.data.copy_(bert_last_layer_attn_in_bias_t)
    
        bert_last_layer_attn_out_weights_t = self.base.encoder.layer[-1].attention.output.dense.weight.data
        bert_last_layer_attn_out_bias_t = self.base.encoder.layer[-1].attention.output.dense.bias.data
        self.head.encoder_attn.out_proj.weight.data.copy_(bert_last_layer_attn_out_weights_t)
        self.head.encoder_attn.out_proj.bias.data.copy_(bert_last_layer_attn_out_bias_t)
                    
        return self

    def save(self, model_path: str):
        self.tokenizer.save_pretrained(model_path)
        self.config.save_pretrained(model_path)
        self.save_pretrained(model_path)
