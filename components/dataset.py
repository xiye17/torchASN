# coding=utf-8
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


class Dataset:
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size *
                                  batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            if shuffle:
                batch_examples.sort(key=lambda e: -len(e.src_toks))

            yield batch_examples


class Example:
    def __init__(self, src_toks, tgt_toks, tgt_ast, idx=0, tgt_actions=None, meta=None):
        self.src_toks = src_toks
        self.tgt_toks = tgt_toks
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta



def sent_lens_to_mask(lens, max_length):
    mask = [[1 if j < l else 0 for j in range(max_length)] for l in lens]
    # match device of input
    return mask

class Batch(object):
    def __init__(self, examples, grammar, vocab, train=True, cuda=True):
        self.examples = examples

        # self.src_sents = [e.src_sent for e in self.examples]
        # self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.cuda = cuda
        self.train = train
        self.build_input()

    def __len__(self):
        return len(self.examples)

    def build_input(self):
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        max_sent_len = max([len(x.src_toks.split()) for x in self.examples])


        self.sents = tokenizer([x.src_toks for x in self.examples], padding=True, truncation=True, max_length=max_sent_len, return_tensors='pt')
        # print(self.sents)
        self.sent_lens = torch.LongTensor([ex.tolist().index(3)+1 for ex in self.sents['input_ids']])

        # print("Cuda: ", self.cuda)
        if self.cuda:
            self.sents = {k : v.cuda() for k, v in self.sents.items()}
            self.sent_lens = self.sent_lens.cuda()

        if self.train:
            print('fdsfsfffsdfsfdsfsgagsgsgfsssssss')
            [self.compute_choice_index(e.tgt_actions) for e in self.examples]

    def compute_choice_index(self, node):
        if isinstance(node, list):
            for sub_node in node:
                self.compute_choice_index(sub_node)

        elif node.action.action_type == "ApplyRule":
            candidate = self.grammar.get_prods_by_type(node.action.type)
            # print(candidate)
            node.action.choice_index = candidate.index(node.action.choice)

            [self.compute_choice_index(x) for x in node.fields]
        elif node.action.action_type == "GenToken":
            token_vocab = self.vocab.primitive_vocabs[node.action.type]
            node.action.choice_index = token_vocab[node.action.choice]
        elif node.action.action_type == "Reduce":
            node.action.choice_index = -2

        else:
            raise ValueError("invalid action type", node.action.action_type)
