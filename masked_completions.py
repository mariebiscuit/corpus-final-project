from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

def inference_loop(masked_sentences: list[str], word_groups_of_interest: dict, 
                   model, tokenizer, batch_size=50):
    
    top5_results = []
    prob_results = {wg: [] for wg in word_groups_of_interest}

    for batch in tqdm(range(0, len(masked_sentences), batch_size)):
        inputs = tokenizer(masked_sentences[batch: batch+batch_size],
                          padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        for wg in word_groups_of_interest.keys():
            words = word_groups_of_interest[wg]
            probsum = torch.zeros((probs.shape[0], len(words)))
            for i, word in enumerate(words):
                token_id = tokenizer.encode(word)[1:-1]
                probsum[:, i] = probs[range(len(mask_token_index)), mask_token_index, token_id]
            
            prob_results[wg] += torch.sum(probsum, dim=1).tolist()


        top_5_tokens = torch.argsort(logits[range(len(mask_token_index)), mask_token_index], -1, descending=True)[:, :5]
        try:
            top5_results += [[tokenizer.decode([i]) for i in row] for row in top_5_tokens]
        except:
            top5_results += ['']


    return top5_results, prob_results

def get_completions(masked_sentences, word_groups_of_interest):
    bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base").to('mps')
    bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    bertweet_tokenizer.padding_side = "left"

    top5_completions, prob_results = inference_loop(masked_sentences, word_groups_of_interest, bertweet, bertweet_tokenizer)

    return top5_completions, prob_results