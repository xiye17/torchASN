
from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.sparql.sparql_transition_system import SparqlTransitionSystem
from models.ASN import ASNParser
from tqdm import tqdm


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    parser = ASNParser.load(args.model_file, ex_args=args)

    to_print = list()
    to_print_target = list()
    grammar = Grammar.from_text(open(args.asdl_file).read())
    transition_system = SparqlTransitionSystem(grammar)

    for dev_set_ in tqdm(test_set.batch_iter(batch_size=1041, shuffle=False)):
        parser.eval()
        with torch.no_grad():
            batch = Batch(dev_set_, parser.grammar, parser.vocab, train=False, cuda=parser.args.cuda)
            sent_embedding = parser.src_embedding(batch.sents)
            parse_results = [(parser.naive_parse(sent_embedding[ex, :, :].unsqueeze(0),
                                                 batch.sent_lens[ex].unsqueeze(0)), dev_set_[ex].tgt_ast) for ex in
                             range(sent_embedding.shape[0])]

            to_print.extend([transition_system.ast_to_surface_code(x[0]) for x in parse_results])
            to_print_target.extend([transition_system.ast_to_surface_code(x[1]) for x in parse_results])


    match_results = [x == y for x, y in zip(to_print, to_print_target)]
    match_acc = sum(match_results) * 1. / len(match_results)
    print('Eval Acc', match_acc)
    print("\n".join(to_print[:10]))
    print("\n".join(to_print_target[:10]))



    with open("output.txt", "w") as f:
       for c in to_print:
           f.write(c + "\n")


if __name__ == '__main__':
    args = parse_args('test')
    test(args)
