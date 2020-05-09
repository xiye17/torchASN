from components.vocab import *


def build_dataset_vocab(examples, transition_system, src_size=5000, code_size=5000, primitive_size=5000, src_cutoff=0, code_cutoff=0, primitive_cutoff=0):
    src_vocab = VocabEntry.from_corpus([e.src_toks for e in examples], size=src_size, freq_cutoff=src_cutoff)

    # primitive_tokens = [map(lambda a: a.action.token,
                                # filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                            # for e in examples]
    # primitive_vocab = VocabEntry.from_corpus(primitive_xwtokens, size=5000, freq_cutoff=0)

    code_tokens = [e.tgt_toks for e in examples]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=code_size, freq_cutoff=0)

    
    type_primitive_tokens = []
    for tree in map(lambda x: x.tgt_actions, examples):
        type_primitive_tokens.extend(extract_primitive_tokens(tree, transition_system))
    
    primitive_entries = {}
    for prim_type in transition_system.grammar.primitive_types:
        type_entry = PrimitiveVocabEntry.from_corpus([[v for t, v in type_primitive_tokens if t == prim_type ]], size=primitive_size, freq_cutoff=primitive_cutoff)
        primitive_entries[prim_type] = type_entry
    
    for t, e in primitive_entries.items():
        print(t, e.word_to_id.keys())
    return DatasetVocab(src_vocab, code_vocab, primitive_entries)

def extract_primitive_tokens(root, transition_system):
    results = []
    def _exract_primitive_tokens_helper(node):
        if transition_system.grammar.is_primitive_type(node.action.type):
            results.append((node.action.type, node.action.choice))
        else:
            [_exract_primitive_tokens_helper(x) for x in node.fields]
    _exract_primitive_tokens_helper(root)
    return results

def cache_action_choice_idx(examples, transition_system, voacb):
    pass
