import sys

sys.path.append('../')
import os
import numpy as np
import json
from tqdm import tqdm
from OFA.ofa import OFATokenizer
import pickle


tokenizer = OFATokenizer.from_pretrained('./OFA')
tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]', '[e_source]']})


data = json.load(open('../WebQA/WebQA_data_first_release/WebQA_train_val_small.json', 'rb'))
keys = list(data.keys())
pbar = tqdm(total=len(keys))
texts_feats = {}
for key in keys:
    pbar.update(1)
    for txt_posfact in data[key]['txt_posFacts']:
        title = txt_posfact['title']
        text = txt_posfact['fact']
        title = tokenizer.tokenize(' ' + title)
        text = tokenizer.tokenize(' ' + text)
        context_text = ['<title>'] + title + ['</title>'] + text
        attn_mask = [1] * len(context_text)
        context_text = tokenizer.convert_tokens_to_ids(context_text)
        texts_feats[txt_posfact['snippet_id']] = {'ids': context_text, 'masks': attn_mask}
    for txt_negfact in data[key]['txt_negFacts']:
        title = txt_negfact['title']
        text = txt_negfact['fact']
        title = tokenizer.tokenize(' ' + title)
        text = tokenizer.tokenize(' ' + text)
        context_text = ['<title>'] + title + ['</title>'] + text
        attn_mask = [1] * len(context_text)
        context_text = tokenizer.convert_tokens_to_ids(context_text)
        texts_feats[txt_negfact['snippet_id']] = {'ids': context_text, 'masks': attn_mask}
pickle.dump(texts_feats, open('../WebQA/WebQA_texts_ofa_feats.pkl', 'wb'))
print('save to', '../WebQA/WebQA_texts_ofa_feats.pkl')