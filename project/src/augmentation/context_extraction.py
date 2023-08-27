import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from nltk.corpus import stopwords

from typing import List, Union
from operator import itemgetter

class SimilarityContextExtractor():
    class SimilarityMatcher():
        def __init__(self, sim_model, tokenizer):
            self.model = sim_model
            self.tokenizer = tokenizer
        
        def match(self, reference: str, candidates: List[str], k: int):
            if len(candidates) == 0:
                return None
        
            with torch.no_grad():
                encs = self.tokenizer([reference] + candidates, add_special_tokens=True, truncation=True, padding='longest', max_length=512, return_tensors='pt')
                encs = {k: v.to(self.model.device) for k, v in encs.items()}
            
                outputs = self.model(**encs)
                question_emb = outputs.last_hidden_state[0].mean(dim=0).unsqueeze(0)            
                candidates_emb = outputs.last_hidden_state[1:].mean(dim=1).squeeze(1)
                
                similarities = (question_emb @ candidates_emb.T).squeeze(0)
                min_k = min(k, len(candidates))
                indices = torch.topk(similarities, min_k)[1]
                indices = indices.tolist()
                return indices

    def __init__(self, ignore_stopwords: bool =True, device: Union[str, torch.device] ="cpu"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        model_name = "roberta-base"
        self.model = RobertaModel(RobertaConfig.from_pretrained(model_name)).to(torch.device(self.device))
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            
        self.matcher = self.SimilarityMatcher(self.model, self.tokenizer)
        
        self.ignore_stopwords = ignore_stopwords
        if self.ignore_stopwords:
            self.stopwords = set(stopwords.words("english")).union(set(stopwords.words("french")))
    
    def __remove_stopwords(self, text: str):
        return " ".join([token for token in text.split() if token not in self.stopwords])
        
    def extract(self, query: str, contexts: List[str], n: int =1, including: List[int] =[]):
        if (len(contexts) == 0) or (len(query) == 0) or (len(contexts[0]) == 0):
            return []
        
        if not(n >= max(1, len(including))):
            return ValueError(f"n must be between 1 and the number of forcefully including contexts, but got {n}")
            
        if len(including) > 0:
            n -= len(including)    
            if n <= 0:
                return [contexts[i] for i in including]
                
        id_to_context = dict(zip(range(len(contexts)), contexts))
        id_to_context_list = list(id_to_context.items())
        keyed_forced_contexts = dict([id_to_context_list[i] for i in including])
        keyed_contexts = {k:v for k, v in id_to_context.items() if k not in set(keyed_forced_contexts.keys())}
            
        clean_query = self.__remove_stopwords(query) if self.ignore_stopwords else query
        clean_contexts = [(self.__remove_stopwords(context) if self.ignore_stopwords else context) for context in keyed_contexts.values()] 
        indices = self.matcher.match(clean_query, clean_contexts, n)
        keyed_selected_contexts = {k:v for i, (k, v) in enumerate(keyed_contexts.items()) if i in indices}
        
        all_selected_keyed_contexts = sorted(list((keyed_forced_contexts | keyed_selected_contexts).items()), key=itemgetter(0), reverse=False)
        extracted_contexts = [context for _, context in all_selected_keyed_contexts]
        return extracted_contexts
