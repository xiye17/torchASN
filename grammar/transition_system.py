# coding=utf-8
from .dsl_ast import *


class Action(object):
    def __init__(self, type_, choice, action_type):
        self.type = type_
        self.choice = choice
        self.action_type = action_type

        # cache choice index among tokens or feasible types
        self.choice_index = -1


class PlaceHolderAction(Action):
    def __init__(self, type_):
        super().__init__(type_, None, 'PlaceHolder')
    
    def __repr__(self):
        return 'PlaceHolder'


class ApplyRuleAction(Action):
    def __init__(self, type_, production):
        # self.production = production
        super().__init__(type_, production, 'ApplyRule')
        self.production = production

    def __hash__(self):
        return hash(self.choice)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.choice == other.choice

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.choice.__repr__()


class GenTokenAction(Action):
    def __init__(self, type_, token):
        super().__init__(type_, token, 'GenToken')
        self.token = token

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
    def __init__(self):
        super().__init__(None, None, 'Reduce')

    def __repr__(self):
        return 'Reduce'


class ActionTree:
    def __init__(self, action, fields=[]):
        self.action = action
        self.fields = fields

    def copy(self):
        return ActionTree(self.action, [f.copy() for f in self.fields])

    def __repr__(self):
        return '%s( %s )' % (self.action.__repr__(), ",".join([x.__repr__() for x in self.fields]))


class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_action_tree(self, ast_tree):
        return self._get_action_tree(self.grammar.root_type, ast_tree)

    def _get_action_tree(self, dsl_type, ast_node):

        if ast_node is None:  # None -> Reduce Actions.
            return ActionTree(ReduceAction())

        # primitive type
        if dsl_type.is_primitive_type():
            assert isinstance(ast_node, str)
            return ActionTree(GenTokenAction(dsl_type, ast_node))

        # composite type
        if isinstance(ast_node, list):  # multiple case
            multiple_field = []

            for node in ast_node:
                if node is None:
                    continue  # Reduce
                action = ApplyRuleAction(dsl_type, node.production)
                node_fields = [self._get_action_tree(x.field.type, x.value) for x in node.fields]

                multiple_field.append(ActionTree(action, node_fields))
            else:
                multiple_field.append(ActionTree(ReduceAction()))

            return multiple_field

        else:  # single / optional case
            action = ApplyRuleAction(dsl_type, ast_node.production)
            fields = [self._get_action_tree(x.field.type, x.value) for x in ast_node.fields]

            return ActionTree(action, fields)
    
    def build_ast_from_actions(self, action_tree):
        if isinstance(action_tree, list):  # cardinality multiple
            multiple_field = [self.build_ast_from_actions(node) for node in action_tree]
            return multiple_field

        if isinstance(action_tree.action, GenTokenAction):
            # choice contains production(DSLProduction) or token (str)
            return action_tree.action.choice

        if isinstance(action_tree.action, ReduceAction):  # Reduce Action -> None
            return None

        if not action_tree.fields:  # Case for constructor without parameters (agg_op = VOID | SUM)
            return AbstractSyntaxTree(action_tree.action.choice)

        production = action_tree.action.choice
        assert len(action_tree.fields) == len(production.constructor.fields)
        return AbstractSyntaxTree(production, realized_fields=[
                RealizedField(cnstr_f, self.build_ast_from_actions(action_f))
                for action_f, cnstr_f in zip(action_tree.fields, production.constructor.fields)
            ])

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError
