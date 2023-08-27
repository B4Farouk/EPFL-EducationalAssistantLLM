# This script is used to generate responses to questions submitted
# in the same format as the provided prompts.json file.

########################
# HOW TO USE THIS SCRIPT
########################

# python3 gen_script_NotAnAGI.py --input_file="../prompts.json" --output_file="answers.json" --model_path="assistant-t5-large-lm" --batch_size=1

#########################
# SCRIPT CODE STARTS HERE
#########################

import argparse
import json
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from trl import AutoModelForSeq2SeqLMWithValueHead
from tqdm import tqdm

########################
# GLOBALS
########################

default_gen_config = {
    # general
    "num_return_sequences": 1,
    "min_length": 16,
    "max_length": 512,
    
    # sampling
    "top_k": 0,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.3,
    
    # against repetition
    "no_repeat_ngram_size": 4,
    "repetition_penalty": 1.5,
    
    # speed-up
    "use_cache": True 
}

SEED = 42
torch.manual_seed(SEED)

#########################
# FUNCTIONS
#########################

def build_parser():
    parser = argparse.ArgumentParser(
        prog="StudentAssistantLLM",
        description="Generate responses to questions submitted in the same format as the provided prompts.json file."
    )
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input file.", nargs=1)
    parser.add_argument("--output_file", "-o", type=str, required=True, help="Path to the output file.", nargs=1)
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the model.", nargs=1)
    parser.add_argument("--batch_size", "-b", type=int, required=False, help="Batch size to run the generation.", nargs=1, default=1)
    return parser

def load_transform_prompts(filename: str):
    with open(filename, "rb") as reader:
        data = json.load(reader)

    def mcq_assembler(question, choices):
        for i, choice in enumerate(choices):
            question += f"\n{i+1}) {choice}"
        return question
                
    def assemble_prompt(datum):
        is_mcq = datum.get("choices", None) is not None
        question = mcq_assembler(datum["question"], datum["choices"]) if is_mcq\
            else datum["question"]
        return f"Question: {question} Answer:"
        
    prompts = [(datum["guid"], assemble_prompt(datum)) for datum in data]
    return prompts
    
def load_model_and_tokenizer(model_path: str, device: torch.device):    
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    return model, tokenizer

def batch_generate(model, tokenizer, prompts):

    # batch tokenization    
    encs = tokenizer.batch_encode_plus(prompts,
                    truncation=True,
                    padding=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt")

    encs = {k:v.to(model.device) for k,v in encs.items()}
    
    # gen config
    gen_config = deepcopy(default_gen_config)
    
    # generation
    generations = model.generate(**encs, **gen_config)
    return generations
    
#########################
# MAIN
#########################

if __name__ == "__main__":
    # parse the arguments
    args = build_parser().parse_args()
    
    # load and transform the prompts from the input file (filename is a string)
    prompts_with_guids = load_transform_prompts(args.input_file[0])
    
    # check if CUDA is available, and set the device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU.")
    else:
        device = torch.device("cpu") 
        print("Using CPU, since CUDA GPU is unavailable.")
    
    # load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path[0], device=device)
    
    # generate the responses
    responses = []

    for start_index in tqdm(range(0, len(prompts_with_guids), args.batch_size[0])):

        batch = prompts_with_guids[start_index:start_index+args.batch_size[0]]
        guids, prompts = zip(*batch)
        
        # generate the responses for the current batch
        generations = batch_generate(model, tokenizer, list(prompts))

        # decode the generations
        generations = [tokenizer.decode(generation, skip_special_tokens=True) for generation in generations]
        
        # create the answer entries from the generations
        batch_responses = [{
            "guid": guid,
            "model_answer": generation
        } for guid, generation in zip(guids, generations)]
        
        # extend the list of responses with the reponses of the current batch
        responses.extend(batch_responses)
    
    # write the reponses
    with open(args.output_file[0], "w") as writer:
        json.dump(responses, writer, indent=2)
    
    
    
    
    