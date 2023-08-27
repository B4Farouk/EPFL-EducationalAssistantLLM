from transformers import Trainer

from datasets import Dataset
import pandas as pd

from typing import List, Dict

def build_finetuning_hf_dataset(data: List[Dict[str, str]], tokenizer, max_input_length: int = 512):
    # set the padding token to be the eos token
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # build the dataset
    df = pd.DataFrame(data)

    df["input"] = df["question"]
    df["label"] = df["answer"]

    # drop columns
    df = df.drop(columns=["question", "answer"])

    # drop index
    df = df.reset_index(drop=True)

    dataset = Dataset.from_pandas(df)
    
    

    # filter out examples that are too long and print the number of dropped examples
    dataset = dataset.filter(lambda x: len(tokenizer(x["input"], padding=False)["input_ids"]) <= max_input_length)
    print("dropped examples: ", len(df) - len(dataset))

    # tokenize input into input_ids and attention_mask
    dataset = dataset.map(lambda x: tokenizer(x["input"], padding=False, max_length=max_input_length, truncation=True), batched=False)

    # tokenize labels into label 
    def tokenize_labels(example):
        example["labels"] = tokenizer(example["label"], padding=False, max_length=max_input_length, truncation=True)["input_ids"]
        return example
    dataset = dataset.map(tokenize_labels, batched=False)

    # drop columns
    dataset = dataset.remove_columns("input")
    dataset = dataset.remove_columns("label")

    # set format
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # split into train and test with 95/5 split
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    return dataset

class ConversationTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
        assert return_outputs == False, "we did not implement return_outputs=True in RewardTrainer.compute_loss"
        
        print("inputs: ", inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        question = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = model.tokenizer.decode(labels[0], skip_special_tokens=True)
        print("question: ", question)
        print("answer: ", answer)
        loss = Trainer.compute_loss(self, model, inputs, return_outputs=False)
        print("loss: ", loss)
        return loss



