import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from components.dataset import Batch
from grammar.transition_system import ApplyRuleAction, GenTokenAction, ActionTree
from grammar.hypothesis import Hypothesis
import numpy as np
import os
from common.config import update_args
from transformers import AutoModel, AutoConfig

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class CompositeTypeModule(nn.Module):
    def __init__(self, args, type, productions):
        super().__init__()
        self.type = type
        self.productions = productions
        # print(productions)
        self.w = nn.Linear(2 * args.enc_hid_size + args.field_emb_size, len(productions))
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.w(x)

    # x b * h
    def score(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)
        return F.log_softmax(self.w(x),1)


class ConstructorTypeModule(nn.Module):
    def __init__(self,  args, production):
        super().__init__()
        self.production = production
        self.n_field = len(production.constructor.fields)
        # print(production.constructor.fields)
        self.field_embeddings = nn.Embedding(len(production.constructor.fields), args.field_emb_size)
        self.w = nn.Linear(2 * args.enc_hid_size + args.field_emb_size, args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)
    
    def update(self, v_lstm, v_state, contexts):
        # v_state, h_n, c_n where 1 * b * h
        # input: seq_len, batch, input_size
        # h_0 of shape (1, batch, hidden_size)
        # v_lstm(, v_state)
        inputs = self.field_embeddings.weight
        inputs = self.dropout(inputs)
        contexts = contexts.expand([self.n_field, -1])
        inputs = self.w(torch.cat([inputs, contexts], dim=1)).unsqueeze(0)
        v_state = (v_state[0].expand(self.n_field, -1).unsqueeze(0).contiguous(), v_state[1].expand(self.n_field, -1).unsqueeze(0).contiguous())

        _, outputs = v_lstm(inputs, v_state)

        hidden_states = outputs[0].unbind(1)
        cell_states = outputs[1].unbind(1)
        # print(outputs[0].shape, outputs[1].shape)
        return list(zip(hidden_states, cell_states))

class PrimitiveTypeModule(nn.Module):
    def __init__(self, args, type, vocab):
        super().__init__()
        self.type = type
        self.vocab = vocab
        # print(vocab.word_to_id)
        self.w = nn.Linear(2 * args.enc_hid_size + args.field_emb_size, len(vocab))

    def forward(self, x):
        return self.w(x)
    # need a score

    # x b * h
    def score(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)
        return F.log_softmax(self.w(x),1)


class ASNParser(nn.Module):
    def __init__(self, args, transition_system, vocab):
        super().__init__()

        # encoder
        self.args = args

        self.src_embedding = EmbeddingLayer(args.src_emb_size, vocab.src_vocab.size(), args.dropout)
        self.encoder = RNNEncoder(args.src_emb_size, args.enc_hid_size, args.dropout, True)
        self.transition_system = transition_system
        self.vocab = vocab
        grammar = transition_system.grammar
        self.grammar = grammar
        # init
        comp_type_modules = []

        for dsl_type in grammar.composite_types:
            # print((dsl_type.name, grammar.get_prods_by_type(dsl_type)))
            comp_type_modules.append((dsl_type.name,
                                      CompositeTypeModule(args, dsl_type, grammar.get_prods_by_type(dsl_type))))
        # print([i for i in comp_type_modules])
        self.comp_type_dict = nn.ModuleDict(comp_type_modules)

        # init
        cnstr_type_modules = []
        for prod in grammar.productions:
            cnstr_type_modules.append((prod.constructor.name,
                                       ConstructorTypeModule(args, prod)))
        self.const_type_dict = nn.ModuleDict(cnstr_type_modules)
        # print(cnstr_type_modules)
        prim_type_modules = []
        for prim_type in grammar.primitive_types:
            prim_type_modules.append((prim_type.name,
                                      PrimitiveTypeModule(args, prim_type, vocab.primitive_vocabs[prim_type])))
        self.prim_type_dict = nn.ModuleDict(prim_type_modules)
        # print(prim_type_modules)
        self.v_lstm = nn.LSTM(args.enc_hid_size, args.enc_hid_size)
        self.attn = LuongAttention(args.enc_hid_size, 2 * args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)

    def score(self, examples):
        batch = Batch(examples, self.grammar, self.vocab, cuda=self.args.cuda)
        sent_embedding = self.src_embedding(batch.sents)

        # scores = [self._score(ex) for ex in examples]
        scores = [self._score(sent_embedding[ex, :, :].unsqueeze(0), batch.sent_lens[ex].unsqueeze(0), examples[ex].tgt_actions) for ex in range(sent_embedding.shape[0])]
        # print(scores)
        return torch.stack(scores)
    

    def _score(self, ex, sent_lens, tgt_actions):
        # for i in self.vocab.primitive_vocabs.values():
        #     print(i.word_to_id)
        # batch = Batch([ex], self.grammar, self.vocab, cuda=self.args.cuda)

        context_vecs, encoder_outputs = self.encode(ex, sent_lens)
        init_state = encoder_outputs
        # print(batch.sent_masks)
        return self._score_node(self.grammar.root_type, init_state, tgt_actions, context_vecs)

    def encode(self, sent_embedding, sent_lens):
        # sent_lens = batch.sent_lens
        # sent
        # print(batch.sents)
        # sent_embedding = self.src_embedding(batch.sents)

        # print(sent_embedding.shape)
        # print(sent_lens)
        context_vecs, final_state = self.encoder(sent_embedding, sent_lens)

        # L * b * hidden,  
        # print(context_vecs.size(), final_state[0].size(), final_state[1].size())
        return context_vecs, final_state

    def _score_node(self, node_type, v_state, action_node, context_vecs):
        v_output = self.dropout(v_state[0])
        contexts = self.attn(v_output.unsqueeze(0), context_vecs).squeeze(0)

        # print(contexts.shape)
        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_output, contexts)
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            score = -1 * scores.view([-1])[action_node.action.choice_index]
            # print("Primitive", score)
            return score


        cnstr = action_node.action.choice.constructor
        # print(cnstr.name)
        # print(node_type.name)
        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_output, contexts)
        score = -1 * scores.view([-1])[action_node.action.choice_index]
        # print("Apply", score)

        # pass through
        # print(node_type.name)
        cnstr_module = self.const_type_dict[cnstr.name]
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
        for next_field, next_state, next_action in zip(cnstr.fields, cnstr_results, action_node.fields):
            # print(next_field, next_state, next_action)
            score += self._score_node(next_field.type, next_state, next_action, context_vecs)
        return score

    def naive_parse(self, sent_embedding, sent_lens):
        # batch = Batch([ex], self.grammar, self.vocab, train=False)
        context_vecs, encoder_outputs = self.encode(sent_embedding, sent_lens)
        init_state = encoder_outputs
        # print("**************************************")
        action_tree = self._naive_parse(self.grammar.root_type, init_state, context_vecs, 1)

        # print(self.transition_system.build_ast_from_actions(action_tree))
        return self.transition_system.build_ast_from_actions(action_tree)

    def _naive_parse(self, node_type, v_state, context_vecs, depth):

        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        #print('-------------------------------')
        #print("nodetype = ", node_type.name)

        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)
        if depth > 9:
            return ActionTree(None)

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            #print('primitive module = ', module)
            # scores = mask * module()
            scores = module.score(v_state[0], contexts).cpu().numpy().flatten()
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            # score = -1 * scores.view([-1])[action_node.action.choice_index]
            choice_idx = np.argmax(scores)
            #print("word = ", module.vocab.get_word(choice_idx))
            return ActionTree(GenTokenAction(node_type, module.vocab.get_word(choice_idx)))

        
        comp_module = self.comp_type_dict[node_type.name]
        #print("comp type = ", comp_module)
        scores = comp_module.score(v_state[0], contexts).cpu().numpy().flatten()
        choice_idx = np.argmax(scores)
        #print("comp_module.productions = ", comp_module.productions)
        production = comp_module.productions[choice_idx]

        #print("production = ", production)
        action = ApplyRuleAction(node_type, production)
        #print("action = ", action)
        cnstr = production.constructor
        #print("constructor.name = ", cnstr.name)
        # pass through
        cnstr_module = self.const_type_dict[cnstr.name]
        # print("cnstr module = ", cnstr_module)
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
        #print("BEGIN recursive")
        action_fields = [self._naive_parse(next_field.type, next_state, context_vecs, depth+1) for next_field, next_state in zip(cnstr.fields, cnstr_results)]
        # print(action)
        #print("action_fields = ", action_fields)
        #print("END recursive")
        #print("-------------------------------")
        return ActionTree(action, action_fields)

    def parse(self, ex):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs

        # action_tree = self._naive_parse(self.grammar.root_type, init_state, context_vecs, batch.sent_masks, 1)
        
        completed_hyps = []
        cur_beam = [Hypothesis.init_hypothesis(self.grammar.root_type, init_state)]
        
        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp(hyp, context_vecs, batch.sent_masks)
                hyp_pools.extend(continuations)
            
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            # next_beam = next_beam[:self.args.beam_size]
            
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            for hyp_i, hyp  in enumerate(hyp_pools[:num_slots]):
                if hyp.is_complete():
                    completed_hyps.append(hyp)
                else:
                    cur_beam.append(hyp)
            
            if not cur_beam:
                break
        
        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps

    def continuations_of_hyp(self, hyp, context_vecs, context_masks):

        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        
        pending_node, v_state = hyp.get_pending_node()
        
        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)

        node_type = pending_node.action.type

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_state[0], contexts).cpu().numpy().flatten()
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            # score = -1 * scores.view([-1])[action_node.action.choice_index]
            # choice_idx = np.argmax(scores)
            continuous = []
            for choice_idx, score in enumerate(scores):
                continuous.append(hyp.copy_and_apply_action(GenTokenAction(node_type, module.vocab.get_word(choice_idx)), score))
                # return ActionTree()
            return continuous

        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_state[0], contexts).cpu().numpy().flatten()

        continuous = []
        for choice_idx, score in enumerate(scores):
            production = comp_module.productions[choice_idx]
            action = ApplyRuleAction(node_type, production)
            cnstr = production.constructor
            # pass through
            cnstr_module = self.const_type_dict[cnstr.name]
            # cnstr_results = const_module.iup()
            # next_states = self.v_lstm( [1 * 1 * x], v_state)
            cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
            continuous.append(hyp.copy_and_apply_action(ApplyRuleAction(node_type, production), score, cnstr_results))
        return continuous

    def save(self, filename):
        # TODO return old version of this method to save dict of params
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(self, filename)

    @classmethod
    def load(cls, model_path, ex_args=None, cuda=False):
        params = torch.load(model_path)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        saved_state = params['state_dict']
        saved_args.cuda = cuda
        if ex_args:
            update_args(saved_args, ex_args)
        parser = cls(saved_args, transition_system, vocab)
        # print([i for i in saved_state][:56])
        parser.load_state_dict(saved_state)
        # parser = torch.load(model_path)
        print("load")

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser

    def forward(self, input):
        return [self.naive_parse(ex) for ex in input]

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        # Compute token embeddings
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
        # config = AutoConfig.from_pretrained("cointegrated/rubert-tiny")  # Download configuration from S3 and cache.
        # self.model = AutoModel.from_config(config)

        for param in self.model.parameters():
            param.requires_grad = True

        # TODO add condition train/test
        # for i in range(24):
        #     for param in self.model.encoder.layer[i].output.parameters():
        #         param.requires_grad = True
        # self.embedding = nn.Embedding(full_dict_size, embedding_dim)
        # self.linear = nn.Linear(1024, embedding_dim)
        self.dropout = nn.Dropout(embedding_dropout_rate)

        # nn.init.uniform_(self.embedding.weight, -1, 1)

    def forward(self, input):

        model_output = self.model(**input)
        # print(model_output[0])
        embeddings = model_output.last_hidden_state
        # embeddings = torch.nn.functional.normalize(embeddings)
        # embedded_words = mean_pooling(model_output, input['attention_mask'])
        # print(embedded_words.shape)
        final_embeddings = self.dropout(embeddings)
        return final_embeddings

class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=self.bidirect)
        self.init_weight()
        self.dropout = nn.Dropout(dropout)

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        # print(embedded_words, input_lens.cpu())

        packed_embedding = nn.utils.rnn.pack_padded_sequence(
            embedded_words, input_lens.cpu(), batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        # print(packed_embedding)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat(
                (c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        # print(max_length, output.size(), h_t[0].size(), h_t[1].size())

        output = self.dropout(output)
        # print(output.shape, h_t[0].shape, h_t[1].shape)
        return (output, h_t)


class LuongAttention(nn.Module):

    def __init__(self, hidden_size, context_size=None):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = hidden_size if context_size is None else context_size
        self.attn = torch.nn.Linear(self.context_size, self.hidden_size)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.attn.weight, gain=1)
        nn.init.constant_(self.attn.bias, 0)

    # input query: batch * q * hidden, contexts: c * batch * hidden
    # output: batch * len * q * c
    def forward(self, query, context, inf_mask=None, requires_weight=False):
        # Calculate the attention weights (energies) based on the given method
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)
        # print(query.shape, context.shape)
        e = self.attn(context)
        # print(e.shape)
        # e: B * Q * C
        # print(query.shape, e.transpose(1, 2).shape)
        e = torch.matmul(query, e.transpose(1, 2))
        if inf_mask is not None:
            e = e + inf_mask.unsqueeze(1)

        # dim w: B * Q * C, context: B * C * H, wanted B * Q * H
        # print(e.shape)
        w = F.softmax(e, dim=2)
        # print(w.shape, context.shape)
        c = torch.matmul(w, context)
        # # Return the softmax normalized probability scores (with added dimension
        if requires_weight:
            return c.transpose(0, 1), w
        # print(c.transpose(0, 1).shape)
        return c.transpose(0, 1)