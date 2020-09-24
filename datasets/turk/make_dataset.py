from os.path import join
import sys
sys.path.append('.')
import numpy as np
import pickle
from grammar.grammar import Grammar

# from grammar.hypothesis import Hypothesis, ApplyRuleAction
# from components.action_info import get_action_infos
from components.dataset import Example
# from components.vocab import VocabEntry, Vocab

from grammar.turk.turk_transition_system import *
from datasets.utils import build_dataset_vocab

def load_dataset(split, transition_system):

    prefix = 'data/turk/'
    src_file = join(prefix, "src-{}.txt".format(split))
    spec_file = join(prefix, "spec-{}.txt".format(split))

    examples = []
    for idx, (src_line, spec_line) in enumerate(zip(open(src_file), open(spec_file))):
        print(idx)
        
        src_line = src_line.rstrip()
        spec_line = spec_line.rstrip()
        src_toks = src_line.split()
        
        spec_toks = spec_line.rstrip().split()
        spec_ast = regex_expr_to_ast(transition_system.grammar, spec_toks)

        # sanity check
        reconstructed_expr = transition_system.ast_to_surface_code(spec_ast)
        print(spec_line, reconstructed_expr)
        assert spec_line == reconstructed_expr

        tgt_action_tree = transition_system.get_action_tree(spec_ast)

        # sanity check 
        ast_from_action = transition_system.build_ast_from_actions(tgt_action_tree)
        assert is_equal_ast(ast_from_action, spec_ast)

        expr_from_hyp = transition_system.ast_to_surface_code(ast_from_action)
        assert expr_from_hyp == spec_line
        # sanity check
        # tgt_action_infos = get_action_infos(src_toks, tgt_actions)
        example = Example(idx=idx,
                        src_toks=src_toks,
                        tgt_actions=tgt_action_tree,
                        tgt_toks=spec_toks,
                        tgt_ast=spec_ast,
                        meta=None)

        examples.append(example)
    return examples


def make_dataset():
    
    grammar = Grammar.from_text(open('data/turk/turk_asdl.txt').read())
    transition_system = TurkTransitionSystem(grammar)

    train_set = load_dataset("train", transition_system)
    dev_set = load_dataset("val", transition_system)
    test_set = load_dataset("test", transition_system)
    # get vocab from actions
    vocab = build_dataset_vocab(train_set, transition_system, src_cutoff=2)
    
    # cache decision using vocab can be done in train
    pickle.dump(train_set, open('data/turk/train.bin', 'wb'))
    pickle.dump(dev_set, open('data/turk/dev.bin', 'wb'))
    pickle.dump(test_set, open('data/turk/test.bin', 'wb'))
    pickle.dump(vocab, open('data/turk/vocab.bin', 'wb'))

if __name__ == "__main__":
    make_dataset()
