from context_extraction import SimilarityContextExtractor

from typing import List, Dict, Any

from datasets import Dataset
import pandas as pd

##############################################################
# PROCESSING FUNCTIONS
##############################################################

def process_ft_dataset(original_dataset: List[Dict[str, Any]], augmented_dataset:List[Dict[str, Any]], 
                       confidence_threshold: int =4, ctx_extractor: SimilarityContextExtractor =None):
    # check arguments
    assert len(original_dataset) > 0, "empty origiinal dataset"
    assert len("augmented_dataset") > 0, "empty augmented dataset"
    assert confidence_threshold in set(range(1, 6))
    
    ################
    # cleaning phase
    ################
     
    def is_correctly_formatted(entry):
        interaction = entry["interaction"]
        roles = set([interaction_entry["role"] for interaction_entry in interaction])
        
        if ("user" not in roles) or ("assistant" not in roles):
            return False
        
        for interaction_entry in interaction:
            if (interaction_entry["role"] != "system") and len(interaction_entry["content"]) == 0:
                return False
        
        # format should be repititions of: system (optional) -> user -> assistant
        prev = ""
        for interaction_entry in interaction:
            if interaction_entry["role"] == "system":
                if prev != "" and prev != "assistant":
                    return False
                prev = "system"
            elif interaction_entry["role"] == "user":
                if prev == "user":
                    return False
                prev = "user"
            elif interaction_entry["role"] == "assistant":
                if prev != "user":
                    return False
                prev = "assistant"
            
        return True

    def argmax_confidence_interactions(dataset):
        # for each question (identified by sol_id), select the entry with the highest confidence
        sol_id_to_entries = {}
        for entry in dataset:
            sol_id = entry["sol_id"]
            sol_id_to_entries[sol_id] = [entry] + ([] if sol_id not in sol_id_to_entries else sol_id_to_entries[sol_id])
            if len(sol_id_to_entries[sol_id]) >= 2:
                sol_id_to_entries[sol_id] = [max(sol_id_to_entries[sol_id], key=lambda x: x["confidence"])]
                
        # each list of entries is now of length 1, containing the entry with the highest confidence
        dataset = [entry_list[0] for entry_list in sol_id_to_entries.values()]        
            
        return dataset
        
    # filter out entries with a confidence below threshold
    # drop unwanted keys in each entry 
    # keep only correctly formatted entries
    # select only the entry with the highest confidence for each question
    original_dataset = [entry for entry in original_dataset if entry["confidence"] >= confidence_threshold]
    original_dataset = [{k:v for k, v in entry.items() if k in set(["interaction", "sol_id", "confidence"])} for entry in original_dataset]
    original_dataset = [entry for entry in original_dataset if is_correctly_formatted(entry)]
    original_dataset = argmax_confidence_interactions(original_dataset)
    original_dataset = [v for entry in original_dataset for k, v in entry.items() if k == "interaction"]
    
    # similarly for the augmented dataset, except it does not have confidence scores (all entries are assumed to be of confidence 5)
    augmented_dataset = [entry for entry in augmented_dataset if is_correctly_formatted(entry)]
    augmented_dataset = [v for entry in augmented_dataset for k, v in entry.items() if k == "interaction"]
    
    # merge together the original and augmented datasets
    dataset = original_dataset + augmented_dataset
            
    #########################
    # contextualization phase
    #########################
    def qa_dict(question, answer):
        return {
            "question": question,
            "answer": answer
        }
        
    def context_from(question, answer):
        if question.strip().startswith("Question:"):
            format_ = "{}\nAnswer: {}"
        else:
            format_ = "Question: {}\nAnswer: {}"
        return format_.format(question, answer)
        
    def contextualize(follow_up_question: str, contexts: List[str]):
        contextualized = "\n".join(contexts)
        
        if follow_up_question.strip().startswith("Question:"):
            format_ = "{}\n{}\nAnswer: "
        else:
            format_ = "{}\nQuestion: {}\nAnswer: "
            
        contextualized = format_.format(contextualized, follow_up_question)
        return contextualized
    
    def process_interaction(interaction):
        # remove system entries
        interaction = [entry for entry in interaction if entry["role"] != "system"]
        
        # initialize a new interaction with the first question and answer pair
        new_interaction = [
            qa_dict(question=context_from(question=interaction[0]["content"], answer=""), answer=interaction[1]["content"])
        ]
        
        # case it's a single round interaction
        if len(interaction) == 2:
            return new_interaction
        
        # case it's a multi-round interaction
        previous_contexts = [context_from(question=interaction[0]["content"], answer=interaction[1]["content"])]
        scoring_indicators = ["scale of 1 to 5", "confidence score", "how confident", "rate your confidence", "are you sure"]
        for entry in interaction[2:]:
            if entry["role"] == "user":
                follow_up_question = entry["content"]
            if entry["role"] == "assistant":
                follow_up_answer = entry["content"]
                
                # extract the contexts that match the follow-up question
                contexts = ctx_extractor.extract(
                    query=follow_up_question, contexts=previous_contexts,
                    n=2, # at most 2 previous contexts shall be selected
                    including=[-1] # always include the last context as one of the selected contexts
                )    
                contextualized_follow_up_question = contextualize(follow_up_question, contexts)            
                
                # append the contextualized follow-up question and the follow-up answer to the new interaction
                new_interaction.append(qa_dict(contextualized_follow_up_question, follow_up_answer))
                
                # update the previous contexts with the new context if the follow-up question is not asking the assistant to score its previous answer
                if not(any(scoring_indicator in follow_up_question for scoring_indicator in scoring_indicators)):
                    previous_contexts.append(context_from(question=follow_up_question, answer=follow_up_answer))
        
        return new_interaction
    
    dataset = [process_interaction(interaction) for interaction in dataset]
    print(f"number of interactions in the dataset = {len(dataset)}")
    
    dataset = [pair for interaction in dataset for pair in interaction]
    
    # print final dataset length
    print(f"number of question/answer pairs in the dataset = {len(dataset)}")
    
    return dataset

##################################################################
# FUNCTION TO BUILD HF DATASET (FOR FINETUNING & PPO & EVALUATION)
##################################################################

def build_hf_dataset(data: List[Dict[str, str]], tokenizer, gen_margin: int =100):
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