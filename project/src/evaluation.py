from evaluate import load
from disco_score import DiscoScorer
import nltk
nltk.download('punkt')

import numpy as np

from collections import namedtuple
from typing import List

from tqdm import tqdm

class Evaluator():
    EvalResult = namedtuple(
        "EvalResult", [
            "ds_focus_nn", "ds_sent_nn",
            "bs_precision", "bs_recall", "bs_f1",
            "entity_graph", "lexical_chain", 
            "rc", "lc"
            ])
    
    def __init__(self, model_type: str ="bert-base-uncased", device_name: str ="cuda"):        
        self.device_name = device_name
        self.model_type = model_type
        
        self.bert_scorer  = load("bertscore")
        self.disco_scorer = DiscoScorer(model_name=self.model_type, device=self.device_name)
      
    def scores(self, generations: List[str], references_per_generation: List[List[str]]):
        results = []
        
        # lowercasing
        generations = [gen.lower() for gen in generations]
        references_per_generation = [[ref.lower() for ref in refs] for refs in references_per_generation]
                
        # compute bert scores
        print("Computing BERT score")
        bert_scores = self.bert_scorer.compute(predictions=generations, references=references_per_generation, 
                                        model_type=self.model_type)
        print("done.")
        
        # compute discobert scores and other scores
        disco_scores_plus= []
        for gen, refs in tqdm(list(zip(generations, references_per_generation)), desc="Computing DiscoBERT scores"):
            try:
              ds_focus = self.disco_scorer.DS_Focus_NN(gen, refs)
            except RuntimeError:
              ds_focus = np.nan
            except ValueError:
              ds_focus = np.nan
            
            try:
              ds_sent = self.disco_scorer.DS_SENT_NN(gen, refs)
            except RuntimeError:
              ds_sent = np.nan
            except ValueError:
              ds_sent = np.nan
            
            try:
              rc = self.disco_scorer.RC(gen, refs)
            except ZeroDivisionError:
              rc = np.nan

            try:
              lc = self.disco_scorer.LC(gen, refs)
            except ZeroDivisionError:
              lc = np.nan

            try:
              ent_graph = self.disco_scorer.EntityGraph(gen, refs)
            except RuntimeError:
              ent_graph = np.nan
            except ValueError:
              ent_graph = np.nan

            try:
              lex_chain = self.disco_scorer.LexicalChain(gen, refs)
            except RuntimeError:
              lex_chain = np.nan
            except ValueError:
              lex_chain = np.nan

            disco_scores_plus_result = {
                  "ds_focus_nn": ds_focus,
                  "ds_sent_nn": ds_sent,
                  "rc": rc,
                  "lc": lc,
                  "entity_graph": ent_graph,
                  "lexical_chain": lex_chain
            }
            disco_scores_plus.append(disco_scores_plus_result)
            
        # sanity check
        n_samples = len(disco_scores_plus)
        assert n_samples == len(bert_scores["precision"])
        assert n_samples == len(bert_scores["recall"])
        assert n_samples == len(bert_scores["f1"])
        
        # build eval result
        for i in tqdm(range(n_samples), desc="Build Evaluation Results"):
            result = Evaluator.EvalResult(
                ds_focus_nn= disco_scores_plus[i]["ds_focus_nn"],
                ds_sent_nn= disco_scores_plus[i]["ds_sent_nn"],
                rc= disco_scores_plus[i]["rc"],
                lc= disco_scores_plus[i]["lc"],
                entity_graph= disco_scores_plus[i]["entity_graph"],
                lexical_chain= disco_scores_plus[i]["lexical_chain"],
                bs_precision= bert_scores["precision"][i],
                bs_recall= bert_scores["recall"][i],
                bs_f1= bert_scores["f1"][i]
            )
            results.append(result)
            
        return results
    
    def evaluate(self, generations: List[str], references_per_generation: List[List[str]], agg_method: str ="mean"):
        assert agg_method in set(["mean", "max", "min", "median", "std"]), f"Invalid aggregation method {agg_method}"
        
        # compute the scores
        results = self.scores(generations, references_per_generation)
        
        # select aggregation function
        agg_fn = None
        if agg_method == "mean":
            agg_fn = np.nanmean
        if agg_method == "max":
            agg_fn = np.max
        if agg_method == "min":
            agg_fn = np.min
        if agg_method == "median":
            agg_fn = np.median
        if agg_method == "std":
            agg_fn = np.std
        
        # aggregate the scores
        return Evaluator.EvalResult(
            ds_focus_nn= agg_fn([result.ds_focus_nn for result in results]),
            ds_sent_nn= agg_fn([result.ds_sent_nn for result in results]),
            rc= agg_fn([result.rc for result in results]),
            lc= agg_fn([result.lc for result in results]),
            entity_graph= agg_fn([result.entity_graph for result in results]),
            lexical_chain= agg_fn([result.lexical_chain for result in results]),
            bs_precision= agg_fn([result.bs_precision for result in results]),
            bs_recall= agg_fn([result.bs_recall for result in results]),
            bs_f1= agg_fn([result.bs_f1 for result in results])
        )

        
        