import trl 
trl.set_seed(42)

from trl import PPOTrainer, PPOConfig
import bitsandbytes as bnb

from datasets import Dataset

from lion_pytorch import Lion

import torch
from torch import nn
torch.manual_seed(42)

from rewards import RewardModel

from tqdm import tqdm
import pandas as pd
import os
from typing import List, Dict, Any

#####################################################################################################
# Insipired by:
        # https://github.com/lvwerra/trl/blob/main/examples/sentiment/notebooks/gpt2-sentiment.ipynb
        # https://github.com/lvwerra/trl/blob/main/examples/toxicity/scripts/gpt-j-6b-toxicity.py
        # https://github.com/jasonvanf/llama-trl/blob/main/tuning_lm_with_rl.py
#####################################################################################################

def build_ppo_hf_dataset(data: List[Dict[str, str]], tokenizer, gen_margin: int =100):
    # build the dataset
    dataset = Dataset.from_pandas(pd.DataFrame(data=data))
    dataset = dataset.rename_column("question", "query")
    
    # prepare the input
    def prepare_query(sample):
        # tokenization
        sample["input_ids"] = tokenizer(
            sample["query"], max_length=tokenizer.model_max_length,
            padding=False, truncation=True, return_attention_mask=True, return_tensors="pt")["input_ids"].squeeze()
        return sample
    # filter out the samples that are too long
    def query_filter(sample):
        return len(sample["input_ids"]) <= (tokenizer.model_max_length - gen_margin)
        
    dataset = dataset.map(prepare_query, batched=False)
    dataset = dataset.filter(query_filter, batched=False)
    dataset.set_format("torch")
    
    dataset = dataset.select_columns(["query", "input_ids"])
    dataset = dataset.shuffle(seed=42)
    
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset["train"], dataset["test"]
    
class PPOTrainerWrapper:
    @staticmethod
    def create_ppo_config(
        model_name: str,
        lr: float =1.41e-5,
        batch_size: int =8,
        ppo_epochs: int =1,
        mini_batch_size: int =4,
        grad_cumul_steps: int =4,
        optimize_cuda_cache: bool =True
        ):
        params = {
            "model_name": model_name,
            
            "batch_size": batch_size,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            
            "learning_rate": lr,
            "gradient_accumulation_steps": grad_cumul_steps,

            "init_kl_coef": 0.2, # initialize the beta coefficient in linear or control KL
            
            "gamma": 1, # discount factor
            "lam": 0.95, # eligibility trace factor?
            
            "adap_kl_ctrl": True, # whether to use control KL
            "target_kl": 0.1, # target KL
            "cliprange": 0.2,
            "cliprange_value": 0.95,
            
            "vf_coef": 0.1,
            "horizon": 10000, # number of times to adapt the beta coefficient in control KL
            
            "optimize_cuda_cache": optimize_cuda_cache,
            
            "log_with": "tensorboard",
            "accelerator_kwargs": {"project_dir": "../../logs/ppo"} 
        }
        return PPOConfig(**params)
    
    @staticmethod
    def ppo_collator(samples):
        batch = {k: [s[k] for s in samples] for k in samples[0].keys()}
        return batch
    
    def __init__(self, 
                 config: PPOConfig, 
                 model: nn.Module, tokenizer: nn.Module, gen_config: Dict[str, Any],
                 reward_model: RewardModel,
                 dataset: Dataset,
                 optim_name: str):
        assert optim_name in ["adam", "adam8bit", "lion"]
        
        self.reward_model = reward_model
        self.gen_config = gen_config
            
        if optim_name == "adam":
            optimizer = torch.optim.AdamW(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate)
        if optim_name == "adam8bit":
            optimizer = bnb.optim.Adam8bit(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate)
        if optim_name == "lion":
            optimizer = Lion(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate / 3 # set the learning rate to 1/3 of the original value used by Adam optimizers, proven to be better
            )
        model.train()
        print("number of optimizable parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("model device:", model.pretrained_model.device)
        print("reward device:", reward_model.device)
        
        self.trainer = PPOTrainer(
            config=config,
            
            model=model,
            tokenizer=tokenizer,
                        
            dataset=dataset,
            data_collator=PPOTrainerWrapper.ppo_collator,
            
            optimizer=optimizer
        )
    
    def generate(self, input_ids: torch.Tensor):
        with torch.no_grad():
            generation = self.trainer.generate(input_ids, return_prompt=False, **self.gen_config).squeeze()
            return generation
    
    def batch_decode(self, token_ids: List[List[int]]):
        decoded = self.trainer.tokenizer.batch_decode(token_ids,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
        return decoded
    
    def compute_rewards(self, queries: List[str], responses: List[str]):
        texts = [f"Human: {query} Assistant: {response}" for query, response in zip(queries, responses)]
        with torch.no_grad():
            encs = self.reward_model.tokenize(texts).to(self.reward_model.device)
            return self.reward_model(**encs)
     
    def train(self, save_path: str, save_freq: int =10):
        for epoch, batch in enumerate(tqdm(self.trainer.dataloader)):
            query_txts = batch["query"]
            query_ts   = [ids.to(self.trainer.model.pretrained_model.device) for ids in batch["input_ids"]]
                
            # for each query, generate a reply and decode it
            self.trainer.model.eval()
            response_ts = [self.generate(input_ids) for input_ids in query_ts]
            response_txts = self.batch_decode(response_ts)
            
            # compute the rewards for each query-response pair
            rewards = [reward for reward in self.compute_rewards(query_txts, response_txts)]
            
            # PPO training for config.ppo_epochs using config.mini_batch_size per epoch
            # where the mini batch is sampled from the batch
            self.trainer.model.train()
            stats = self.trainer.step(
                queries=query_ts,
                responses=response_ts,
                scores=rewards
            )
            
            # logging
            self.trainer.log_stats(stats, batch, rewards)
            
            # saving
            if epoch % save_freq == 0:
                if self.trainer.accelerator.is_main_process:
                    # create a folder for the current epoch if it does not exist
                    epoch_save_path = save_path+f"/ep{epoch:03d}"
                    self.__save(epoch_save_path)
        
        last_save_path = save_path+"/last"
        self.__save(last_save_path)
        
    def __save(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the model and the tokenizer
        self.trainer.accelerator.unwrap_model(self.trainer.model).save_pretrained(save_path)
        self.trainer.tokenizer.save_pretrained(save_path)
