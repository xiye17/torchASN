from os.path import join
import sys

sys.path.append('.')

import pickle
from components.dataset import Example

from grammar.grammar import Grammar
from grammar.sparql.sparql_transition_system import *
from datasets.utils import build_dataset_vocab
sys.path.append('.')


def load_dataset(split, transition_system):
    """
    @split: str - datasplit type, a.g.: train/test/val
    """

    prefix = 'data/sparql/'
    src_file = join(prefix, "src-{}.txt".format(split))  # Input text
    spec_file = join(prefix, "spec-{}.txt".format(split))  # Desired code

    examples = []

    for idx, (src_line, spec_line) in enumerate(zip(open(src_file, encoding="utf8"), open(spec_file, encoding="utf8"))):

        src_line = src_line.rstrip()
        spec_line = spec_line.rstrip()

        # src_toks = src_line.split()
        src_toks = src_line

        spec_toks = spec_line.split()

        # Parse code to AST
        spec_ast = sparql_expr_to_ast(transition_system.grammar, spec_toks)

        # sanity check
        reconstructed_expr = transition_system.ast_to_surface_code(spec_ast)
        assert spec_line == reconstructed_expr

        # Parse ADSL_AST to ActionTree
        tgt_action_tree = transition_system.get_action_tree(spec_ast)

        # sanity check
        ast_from_action = transition_system.build_ast_from_actions(tgt_action_tree)
        # generated_code = transition_system.ast_to_surface_code(ast_from_action)
        # target_code = transition_system.ast_to_surface_code(spec_ast)
        assert is_equal_ast(ast_from_action, spec_ast)

        example = Example(idx=idx,
                          src_toks=src_toks,
                          tgt_actions=tgt_action_tree,
                          tgt_toks=spec_toks,
                          tgt_ast=spec_ast,
                          meta=None)

        examples.append(example)
    return examples


def make_dataset():
    
    grammar = Grammar.from_text(open('data/sparql/sparql_asdl.txt').read())
    transition_system = SparqlTransitionSystem(grammar)

    train_set = load_dataset("train", transition_system)
    dev_set = load_dataset("val", transition_system)
    test_set = load_dataset("test", transition_system)
    # get vocab from actions
    vocab = build_dataset_vocab(train_set, transition_system, src_cutoff=2)
    # cache decision using vocab can be done in train
    pickle.dump(train_set, open('data/sparql/train.bin', 'wb'))
    pickle.dump(dev_set, open('data/sparql/dev.bin', 'wb'))
    pickle.dump(test_set, open('data/sparql/test.bin', 'wb'))
    pickle.dump(vocab, open('data/sparql/vocab.bin', 'wb'))


if __name__ == "__main__":
    make_dataset()
