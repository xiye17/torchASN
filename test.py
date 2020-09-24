import sys
from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.turk.turk_transition_system import TurkTransitionSystem
from models.ASN import ASNParser
from models import nn_utils

from torch import optim
import os

import subprocess
from tqdm import tqdm

def post_process(x):
    x = x.replace("<m0>", "<!>")
    x = x.replace("<m1>", "<@>")
    x = x.replace("<m2>", "<#>")
    x = x.replace("<m3>", "<$>")
    x = x.replace(" ", "")
    return x

def check_equiv(spec0, spec1):
    if spec0 == spec1:
        # print("exact", spec0, spec1)
        return True
    # try:
    out = subprocess.check_output(
        ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'equiv',
            spec0, spec1], stderr=subprocess.DEVNULL)
    out = out.decode("utf-8")
    out = out.rstrip()
    # if out == "true":
    #     print("true", spec0, spec1)

    return out == "true"

def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    parser = ASNParser.load(args.model_file, ex_args=args)    

    parser.eval()
    with torch.no_grad():
        parse_results = []
        for ex in tqdm(test_set, desc='Decoding', file=sys.stdout, total=len(test_set)):
            parse_results.append(parser.parse(ex) )
    # match_results = [ parser.transition_system.compare_ast(e.tgt_ast, r) for e, r in zip(test_set, parse_results)]
    # match_acc = sum(match_results) * 1. / len(match_results)
    # print("Eval Acc", match_acc)bv
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    top_asts = [ act_tree_to_ast(x[0].action_tree) if x else None for x in parse_results]
    top_codes = [parser.transition_system.ast_to_surface_code(x) for x in top_asts]
    # match_results = [ parser.transition_system.compare_ast(e.tgt_ast, r) for e, r in zip(test_set, top_asts)]
    match_results = [ " ".join(e.tgt_toks) == r for e, r in zip(test_set, top_codes)]
    # top_asts = [parser.transition_system]

    match_acc = sum(match_results) * 1. / len(match_results)
    # [print("%s\n\t==>%s\n\t==>%s" % (" ".join(e.src_toks), " ".join(e.tgt_toks), c)) for e,c in zip(test_set, top_codes)]
    
    with open("output.txt", "w") as f:
        for c in top_codes:
            f.write(c.replace(" ","") + "\n")

    oracle_res = []
    i = 0
    acc = 0
    for e, c in zip(test_set, top_codes):
        gt_code = post_process(" ".join(e.tgt_toks))
        pred_code = post_process(c)
        eq_res = check_equiv(pred_code, gt_code)
        oracle_res.append(eq_res)
        acc += eq_res
        i += 1
        # print(acc, i)
    print("String Acc", match_acc)
    print("DFA Acc", sum(oracle_res) * 1.0/len(oracle_res) )

if __name__ == '__main__':
    args = parse_args('test')
    test(args)
