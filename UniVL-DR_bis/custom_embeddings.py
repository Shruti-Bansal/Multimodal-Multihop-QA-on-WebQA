import json
from visual import TSVFile
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
#import dill as pickle
import math
import base64
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import clip
from data import WebQADataset
from gen_embeddings import TextDataset, gen_embeddings

logger = logging.getLogger()
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_queries(path, qid_set):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            qid = str(example['qid'])
            if qid in qid_set:
                data.append([qid, example['Q']])
    return data

def load_caps(path, imgid_set):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            if imgid in imgid_set:
                data[imgid] = example['caption']
    return data

def load_docs(path, docid_set):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            if did in docid_set:
                data.append([did, example['fact']])
                #data[did] = example['fact']
    return data

class ImgDataset(Dataset):
    def __init__(self, args, preprocess, imgid_set, captions=None):
        self.max_seq_len = args.max_seq_len

        self.img_map = {}
        self.img_ids = []
        self.captions = captions
        self.preprocess_fn = preprocess
        self.imgid_set = imgid_set

        all_img_num = 0
        with open(args.img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                if str(tokens[0]) in self.imgid_set:
                    all_img_num += 1
                    self.img_map[tokens[0]] = int(tokens[1])
                    self.img_ids.append(tokens[0])
        self.img_tsv = TSVFile(args.img_feat_path, all_img_num)

    def __len__(self):
        return len(self.img_ids)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(Image.open(io.BytesIO(base64.b64decode(img))))
        if self.captions != None:
            cap = self.captions[idx]
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        img_caps = []
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            if 'img_caps' in example:
                img_caps.append(example['img_caps'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
        if len(img_caps) != 0:
            processed_batch['img_caps'] = clip.tokenize(img_caps)#truncate=True
        return processed_batch

    def __getitem__(self, index):
        img_idx = self.img_ids[index]
        img_inputs = self.encode_img(img_idx)
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if 'cap' in img_inputs:
            instance['img_caps'] = img_inputs['cap']

        return instance


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--max_seq_len", type=int, default=77)

    parser.add_argument("--out_path", type=str, default="test/")
    parser.add_argument("--checkpoint", type=str, default = "checkpoint_multi_hn/model.best.pt")
    parser.add_argument("--img_feat_path", type=str, default = "../../WebQA/imgs.tsv")
    parser.add_argument("--img_linelist_path", type=str, default = "../data/imgs.lineidx.new")

    parser.add_argument("--query_path", type=str, default="../data/test.json")
    parser.add_argument("--doc_path", type=str, default="../data/all_docs.json")
    parser.add_argument("--cap_path", type=str, default="../data/all_imgs_exp.json")

    parser.add_argument('--encode_txt', action='store_true', default=True)
    parser.add_argument('--encode_img', action='store_true', default=True)
    parser.add_argument('--encode_query', action='store_true', default=True)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()


    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    checkpoint['model']["input_resolution"] = model.input_resolution  # default is 224
    checkpoint['model']["context_length"] = model.context_length  # default is 77
    checkpoint['model']["vocab_size"] = model.vocab_size
    model.load_state_dict(checkpoint['model'])
    #model.cuda()

    #WHAT WE WANT TO EMBED
    qid_set = set()
    imgid_set = set()
    docid_set = set()

    def load_file(path):
        result = []
        with open(path, 'r') as file:
            for line in file:
                result.append(json.loads(line))
        return result
    path_test = "/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/data/test.json"
    test_data = load_file(path_test)
    for query in test_data[:2]:
        qid_set.add(str(query["qid"]))
        imgid_set |= set(query["img_posFacts"])
        imgid_set |= set(query["img_negFacts"])
        docid_set |= set(query["txt_negFacts"])
        docid_set |= set(query["txt_posFacts"])

    if args.encode_query:
        queries = load_queries(args.query_path, qid_set)
        query_data = TextDataset(queries, args.max_seq_len)
        query_sampler = SequentialSampler(query_data)
        query_reader = DataLoader(dataset=query_data, sampler=query_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=query_data.Collector)

        output = os.path.join(args.out_path, 'query_embedding.pkl')
        gen_embeddings(model, query_reader, output)

    if args.encode_img:
        captions = None
        if args.cap_path:
            captions = load_caps(args.cap_path, imgid_set)
        img_data = ImgDataset(args, preprocess, imgid_set, captions=captions)
        sampler = SequentialSampler(img_data)
        img_reader = DataLoader(dataset=img_data, sampler=sampler,
                                      batch_size=args.batch_size, collate_fn=img_data.Collector)
        output = os.path.join(args.out_path, 'img_embedding.pkl')
        gen_embeddings(model, img_reader, output)

    if args.encode_txt:
        docs = load_docs(args.doc_path, docid_set)
        txt_data = TextDataset(docs, args.max_seq_len)
        txt_sampler = SequentialSampler(txt_data)
        txt_reader = DataLoader(dataset=txt_data, sampler=txt_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=txt_data.Collector)
        output = os.path.join(args.out_path, 'txt_embedding.pkl')
        gen_embeddings(model, txt_reader, output)