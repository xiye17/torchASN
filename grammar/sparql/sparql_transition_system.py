# coding=utf-8

from grammar.transition_system import TransitionSystem
from grammar.dsl_ast import RealizedField, AbstractSyntaxTree

"""
SPARQL Transition system.
This code contains functions that implements:
1. Code to ASDL tree parsing
2. ASDL tree code processing
3. ASDL trees comparison
"""


def build_ast_from_toks(grammar, token_subset, rule):
    children = []
    production = grammar.get_prod_by_ctr_name(rule)

    if rule in ["QUERY"]:
        # SELECT
        select_tokens = token_subset[1:token_subset.index('where')]

        select_tokens = " ".join(select_tokens)

        select_node = build_ast_from_toks(grammar, select_tokens, 'SELECT')
        select_field = RealizedField(production['select'], select_node)
        children.append(select_field)

        # WHERE
        where_tokens = token_subset[token_subset.index('where'):token_subset.index('}') + 1]

        # ["where", "{", "SUBJ_1", "dr:locatedinclosest", "?dummy", "." "?dummy", "dr:date_country_end", "?x0", ".", "}"]
        # -> ["SUBJ_1 dr:locatedinclosest?dummy . ?dummy dr:date_country_end ?x0"]
        where_tokens = " ".join(where_tokens[2:-2])

        where_node = build_ast_from_toks(grammar, where_tokens, "WHERE")

        where_field = RealizedField(production['where'], where_node)
        children.append(where_field)

        # ORDER BY
        order_tokens = token_subset[token_subset.index('}') + 1:]

        order_node = build_ast_from_toks(grammar, order_tokens, "ORDERBY")
        order_field = RealizedField(production['order'], order_node)
        children.append(order_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["SELECT"]:
        token_subset = token_subset.strip().split(",")
        select_nodes = [build_ast_from_toks(grammar, select_val, "SELECT_COL") for select_val in token_subset]
        select_nodes.append(None)  # Reduce

        select_field = RealizedField(production['select_val'], select_nodes)
        children.append(select_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["SELECT_COL"]:
        token_subset = token_subset.strip().split(" ")

        # 1. ["count", "(", "distinct", "?x0", ")"]
        # 2. ["count", "(", "?x0", ")"]
        # 3. ["distinct", "?x0"]
        # 4. ["?x0"]

        # agg_op
        agg_op_token = token_subset[0]
        if agg_op_token in ["count", "min", "max", "avg", "sum"]:
            rule = agg_op_token.upper()
            agg_flag = True
            agg_production = grammar.get_prod_by_ctr_name(rule)
            agg_node = AbstractSyntaxTree(agg_production, None)
        else:
            agg_flag = False
            agg_node = None

        agg_field = RealizedField(production['agg_op_val'], agg_node)
        children.append(agg_field)

        # col_type
        if "distinct" in token_subset:
            column_idx = 3 if agg_flag else 1  # case 1 and 2
            rule = "DISTINCT"
        else:
            column_idx = 2 if agg_flag else 0  # case 3 and 4
            rule = "CONST"

        col_type_production = grammar.get_prod_by_ctr_name(rule)
        col_type_field = RealizedField(col_type_production['column'], str(token_subset[column_idx]))
        col_type_node = AbstractSyntaxTree(col_type_production, col_type_field)

        select_col = RealizedField(production['column_type'], col_type_node)
        children.append(select_col)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["WHERE"]:
        token_subset = token_subset.strip().split(".")
        where_nodes = [build_ast_from_toks(grammar, where_val, "WHERE_COL") for where_val in token_subset]
        where_nodes.append(None)  # Reduce

        where_field = RealizedField(production['where_val'], where_nodes)
        children.append(where_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["WHERE_COL"]:
        # ["SUBJ_1", "dr:locatedinclosest", "?dummy"]
        field = token_subset.strip().split(" ")

        subject, predicate, object_ = field

        children.extend([RealizedField(production["subject_val"], str(subject)),
                         RealizedField(production["predicate_val"], str(predicate)),
                         RealizedField(production["object_val"], str(object_))
                         ])

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["ORDERBY"]:
        column_idx = 2

        # 1. ['order', 'by', 'desc', '(', '?x0', ')']
        # 2. ['order', 'by', '?x0']
        # 3. []

        if token_subset:
            if "desc" in token_subset:
                rule = "DESC"
                desc_production = grammar.get_prod_by_ctr_name(rule)
                desc_flag = AbstractSyntaxTree(desc_production, None)
                column_idx = 4
            else:
                desc_flag = None

            desc_field = RealizedField(production['desc_flag'], desc_flag)
            children.append(desc_field)

            token = token_subset[column_idx]
            ord_column = RealizedField(production["ord_column"], token)
            children.append(ord_column)

            ast_node = AbstractSyntaxTree(production, children)
            return ast_node

        return None


def build_sparql_expr_from_ast(sparql_ast):
    tokens = ['select']
    select, where, order = [x.value for x in sparql_ast.fields]

    # SELECT
    for select_col in select.fields[0].value:
        if select_col is None:
            continue  # Reduce skipping

        aggrigation_field, col_type_field = [x.value for x in select_col.fields]

        # agg_op_val field

        if aggrigation_field is not None:
            agg_op = aggrigation_field.production.constructor.name

            tokens.extend([agg_op.lower(), "(", ])
            agg_flag = True

        else:
            agg_flag = False

        # column_type field
        col_type = col_type_field.production.constructor.name

        if col_type == "DISTINCT":
            tokens.append(col_type.lower())

        # col_type -> column
        column_id = col_type_field.fields[0].value
        tokens.append("<unk>" if column_id is None else column_id.lower())

        if agg_flag:
            tokens.append(")")

        tokens.append(",")

    else:
        tokens.pop(-1)  # removes last ","

    # WHERE
    tokens.append("where {")

    for where_col in where.fields[0].value:
        if where_col is None:
            continue  # reduce skipping

        tokens.extend([x.value for x in where_col.fields])
        tokens.append('.')

    tokens.append('}')

    # ORDERBY
    if order is not None:
        order_fields = order.fields
        tokens.append("order by")

        if order_fields[0] is not None:
            tokens.append("desc")

        tokens.extend(["(", order_fields[1].value, ")"])

    return tokens


def sparql_expr_to_ast(grammar, sparql_tokens):
    sparql_ast = build_ast_from_toks(grammar, sparql_tokens, rule="QUERY")
    return sparql_ast


def ast_to_sparql_expr(sparql_ast):
    tokens = build_sparql_expr_from_ast(sparql_ast)
    return " ".join(tokens)

# neglet created time


def is_equal_ast(this_ast, other_ast):
    if not isinstance(other_ast, this_ast.__class__):
        return False

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


class SparqlTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def ast_to_surface_code(self, sparql_ast):
        return ast_to_sparql_expr(sparql_ast)

    def surface_code_to_ast(self, code):
        return sparql_expr_to_ast(self.grammar, code)

    def tokenize_code(self, code, mode):
        raise NotImplementedError
