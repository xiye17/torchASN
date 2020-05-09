# coding=utf-8
from .turk_utils import * 

from grammar.transition_system import TransitionSystem

try:
    from cStringIO import StringIO
except:
    from io import StringIO

from grammar.grammar import *
from grammar.dsl_ast import RealizedField, AbstractSyntaxTree
# from common.registerable import Registrable
"""
# define primitive fields
int, cc, tok

regex = Not(regex arg)
    | Star(regex arg)
    | Concat(regex left, regex right)
    | Or(regex left, regex right)
    | And(regex left, regex right)
    | StartWith(regex arg)
    | EndWith(regex arg)
    | Contain(regex arg)
    | RepAtleast(regex arg, int k)
    | CharClass(cc arg)
    | Const(tok arg)
"""

_NODE_CLASS_TO_RULE = {
    "not": "Not",
    "star": "Star",
    "startwith": "StartWith",
    "endwith": "EndWith",
    "contain": "Contain",
    "concat": "Concat",
    "and": "And",
    "or": "Or",
    "repeatatleast": "RepeatAtleast"
}

def regex_ast_to_asdl_ast(grammar, reg_ast):
    if reg_ast.children:
        rule = _NODE_CLASS_TO_RULE[reg_ast.node_class]
        prod = grammar.get_prod_by_ctr_name(rule)
        # unary
        if rule in ["Not", "Star", "StartWith", "EndWith", "Contain"]:
            child_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[0])
            ast_node = AbstractSyntaxTree(prod,
                                            [RealizedField(prod['arg'], child_ast_node)])
            return ast_node
        elif rule in ["Concat", "And", "Or"]:
            left_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[0])
            right_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[1])
            ast_node = AbstractSyntaxTree(prod,
                                            [RealizedField(prod['left'], left_ast_node),
                                            RealizedField(prod['right'], right_ast_node)])
            return ast_node
        elif rule in ["RepeatAtleast"]:
            # primitive node
            # RealizedField(prod['predicate'], value=node_name)
            child_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[0])
            int_real_node = RealizedField(prod['k'], str(reg_ast.params[0]))
            ast_node = AbstractSyntaxTree(prod, [RealizedField(prod['arg'], child_ast_node), int_real_node])
            return ast_node
        else:
            raise ValueError("wrong node class", reg_ast.node_class)
    else:
        if reg_ast.node_class in ["<num>", "<let>", "<vow>", "<low>", "<cap>", "<any>"]:
            rule = "CharClass"
        elif reg_ast.node_class in ["<m0>", "<m1>", "<m2>", "<m3>"]:
            rule = "Const"
        else:
            raise ValueError("wrong node class", reg_ast.node_class)
        prod = grammar.get_prod_by_ctr_name(rule)
        return AbstractSyntaxTree(prod, [RealizedField(prod['arg'], reg_ast.node_class)])

def regex_expr_to_ast(grammar, reg_tokens):
    reg_ast = build_regex_ast_from_toks(reg_tokens, 0)[0]
    assert reg_ast.tokenized_logical_form() == reg_tokens
    return regex_ast_to_asdl_ast(grammar, reg_ast)

def asdl_ast_to_regex_ast(asdl_ast):
    rule = asdl_ast.production.constructor.name
    if rule in ["CharClass", "Const"]:
        return RegexNode(asdl_ast['arg'].value)
    elif rule in ["Not", "Star", "StartWith", "EndWith", "Contain", "Concat", "And", "Or"]:
        node_class = rule.lower()
        return RegexNode(node_class, [asdl_ast_to_regex_ast(x.value) for x in asdl_ast.fields])
    elif rule in ["RepeatAtleast"]:
        node_class = rule.lower()
        if asdl_ast['k'].value.isdigit():
            param = int(asdl_ast['k'].value)
            child_node = asdl_ast_to_regex_ast(asdl_ast['arg'].value)
            return RegexNode(node_class, [child_node], [param])
        else:
            return RegexNode("none")
    else:
        raise ValueError("wrong ast rule", rule)

def asdl_ast_to_regex_expr(asdl_ast):
    reg_ast = asdl_ast_to_regex_ast(asdl_ast)
    return " ".join(reg_ast.tokenized_logical_form())

# neglet created time
def is_equal_ast(this_ast, other_ast):
    if not isinstance(other_ast, this_ast.__class__):
        return False
    # print(this_ast, other_ast)

    if isinstance(this_ast, AbstractSyntaxTree):
        if this_ast.production != other_ast.production:
            return False

        if len(this_ast.fields) != len(other_ast.fields):
            return False
        for this_f, other_f in zip(this_ast.fields, other_ast.fields):
            if not is_equal_ast(this_f.value, other_f.value):
                return False
        return True
    else:
        return this_ast == other_ast

# @Registrable.register('regex')
class TurkTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def ast_to_surface_code(self, asdl_ast):
        return asdl_ast_to_regex_expr(asdl_ast)
    
    def surface_code_to_ast(self, code):
        return regex_expr_to_ast(self.grammar, code)
    
    def hyp_correct(self, hype, example):
        return is_equal_ast(hype.tree, example.tgt_ast)
    
    def tokenize_code(self, code, mode):
        return code.split()
    
    # def get_primitive_field_actions(self, realized_field):
    #     assert realized_field.cardinality == 'single'
    #     if realized_field.value is not None:
    #         return [GenTokenAction(realized_field.value)]
    #     else:
    #         return []
