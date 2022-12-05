# coding=utf-8
from .sparql_utils import *

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
unk_attr, predicate, sub, obj, order_d

regex = OrderedQuery(select_stmt arg, where_stmt arg1, order_stmt arg2)
    | UnorderedQuery(select_stmt arg, where_stmt arg1)

select_stmt = Select(col_stmt arg)

col_stmt = Sum(col_stmt1)
    | Count(col_stmt1)
    | Avg(col_stmt1)

col_stmt1 = DistinctConst(unk_attr k)
    | Const(unk_attr k)

where_stmt = R0(triplets arg)

triplets = R0(triplets arg) | Trip(triplet_sub left, predicate con, triplet_obj)

triplet_sub = ConstSub(sub k) | Const(unk_attr k)

triplet_obj = ConstObj(obj k) | Const(unk_attr k)

order_stmt = Order(unk_attr col, order_d d)
"""

_NODE_CLASS_TO_RULE = {
    "orderedquery": "OrderedQuery",
    "unorderedquery": "UnorderedQuery",
    "select": "Select",
    "sum": "Sum",
    "count": "Count",
    "avg": "Avg",
    "distinctconst": "DistinctConst",
    "const": "Const",
    "r0" : "R0",
    "trip" : "Trip",
    "r1": "R1",
    "trip1": "Trip1",
    "constsub" : "ConstSub",
    "constobj" : "ConstObj",
    "nothing" : "Nothing"
}

def regex_ast_to_asdl_ast(grammar, reg_ast):
    # print(reg_ast)
    rule = _NODE_CLASS_TO_RULE[reg_ast.node_class]
    # print(grammar._constructor_production_map)
    prod = grammar.get_prod_by_ctr_name(rule)
    # unary
    if rule in ["Sum", "Count", "Avg", "Nothing"]:
        child_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children)
        ast_node = AbstractSyntaxTree(prod,
                                        [RealizedField(prod['arg'], child_ast_node)])
        return ast_node
    elif rule in ["UnorderedQuery"]:
        left_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[0])
        right_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[1])

        ast_node = AbstractSyntaxTree(prod,
                                        [RealizedField(prod['left'], left_ast_node),
                                        RealizedField(prod['right'], right_ast_node)])
        return ast_node
    elif rule in ["OrderedQuery"]:
        left_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[0])
        middle_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children[1])
        param_left_ast_node = RealizedField(prod['col'], str(reg_ast.params[1]))
        param_right_ast_node = RealizedField(prod['d'], str(reg_ast.params[0]))
        ast_node = AbstractSyntaxTree(prod,
                                        [RealizedField(prod['left'], left_ast_node),
                                         RealizedField(prod['middle'], middle_ast_node),
                                        param_left_ast_node,
                                        param_right_ast_node])
        return ast_node
    elif rule in ["Trip", "Trip1"]:
        left_ast_node = RealizedField(prod['left'], str(reg_ast.params[0]))
        int_real_node = RealizedField(prod['k'], str(reg_ast.params[1]))
        right_ast_node = RealizedField(prod['right'], str(reg_ast.params[2]))

        ast_node = AbstractSyntaxTree(prod,
                                        [left_ast_node,
                                         int_real_node,
                                         right_ast_node])
        return ast_node
    elif rule in ["R0", "R1"]:
        left_ast_node = RealizedField(prod['left'], str(reg_ast.params[0]))
        int_real_node = RealizedField(prod['k'], str(reg_ast.params[1]))
        right_ast_node = RealizedField(prod['right'], str(reg_ast.params[2]))
        second_ast_node = regex_ast_to_asdl_ast(grammar, reg_ast.children)

        ast_node = AbstractSyntaxTree(prod,
                                      [left_ast_node,
                                       int_real_node,
                                       right_ast_node,
                                         RealizedField(prod['second'], second_ast_node)])
        return ast_node
    elif rule in ["DistinctConst", "Const"]:
        int_real_node = RealizedField(prod['k'], str(reg_ast.params))
        ast_node = AbstractSyntaxTree(prod, int_real_node)
        return ast_node
    else:
        raise ValueError("wrong node class", reg_ast.node_class)

def regex_expr_to_ast(grammar, reg_tokens):
    reg_ast = build_regex_ast_from_toks(reg_tokens, 0)
    # assert reg_ast.tokenized_logical_form() == reg_tokens
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
    #print(this_ast, other_ast)

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
class SparqlTransitionSystem(TransitionSystem):
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
