# coding=utf-8
from .dsl_ast import *

class Action(object):
    def __init__(self, type, choice, action_type):
        self.type = type
        self.choice = choice
        self.action_type = action_type

        # cache choice index among tokens or feasible types
        self.choice_index = -1

class PlaceHolderAction(Action):
    def __init__(self, type):
        super().__init__(type, None, 'PlaceHolder')
    
    def __repr__(self):
        return 'PlaceHoder'

class ApplyRuleAction(Action):
    def __init__(self, type, production):
        # self.production = production
        super().__init__(type, production, 'ApplyRule')
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
    def __init__(self, type, token):
        super().__init__(type, token, 'GenToken')
        self.token = token

    def __repr__(self):
        return 'GenToken[%s]' % self.token


class ReduceAction(Action):
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

    # def get_actions(self, asdl_ast):
    #     """
    #     generate action sequence given the ASDL Syntax Tree
    #     """

    #     actions = []

    #     parent_action = ApplyRuleAction(asdl_ast.production)
    #     actions.append(parent_action)

    #     for field in asdl_ast.fields:
    #         # is a composite field
    #         if self.grammar.is_composite_type(field.type):
    #             if field.cardinality == 'single':
    #                 field_actions = self.get_actions(field.value)
    #             else:
    #                 field_actions = []

    #                 if field.value is not None:
    #                     if field.cardinality == 'multiple':
    #                         for val in field.value:
    #                             cur_child_actions = self.get_actions(val)
    #                             field_actions.extend(cur_child_actions)
    #                     elif field.cardinality == 'optional':
    #                         field_actions = self.get_actions(field.value)

    #                 # if an optional field is filled, then do not need Reduce action
    #                 if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
    #                     field_actions.append(ReduceAction())
    #         else:  # is a primitive field
    #             field_actions = self.get_primitive_field_actions(field)

    #             # if an optional field is filled, then do not need Reduce action
    #             if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
    #                 # reduce action
    #                 field_actions.append(ReduceAction())

    #         actions.extend(field_actions)

    #     return actions


    def get_action_tree(self, ast_tree):
        # if isinstance(ast_node, str)
        return self._get_action_tree(self.grammar.root_type, ast_tree)

    def _get_action_tree(self, dsl_type, ast_node):
        if dsl_type.is_primitive_type():
            assert isinstance(ast_node, str)
            # print(ast_node, type(ast_node))
            return ActionTree(GenTokenAction(dsl_type, ast_node))

        # primitive type
        # ast_node.value
        action = ApplyRuleAction(dsl_type, ast_node.production)
        fields = [self._get_action_tree(x.field.type, x.value) for x in ast_node.fields]
        # composite type
        return ActionTree(action, fields)
    
    def build_ast_from_actions(self, action_tree):
        if action_tree.action is None: # TODO for now only
            return None

        if not action_tree.fields:
            return action_tree.action.choice
        
        production = action_tree.action.choice
        assert len(action_tree.fields) == len(production.constructor.fields)
        
        return AbstractSyntaxTree(production, realized_fields=[
                RealizedField(cnstr_f, self.build_ast_from_actions(action_f))
                for action_f, cnstr_f in zip (action_tree.fields, production.constructor.fields)
            ])

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    # def get_valid_continuation_types(self, hyp):
    #     if hyp.tree:
    #         if self.grammar.is_composite_type(hyp.frontier_field.type):
    #             if hyp.frontier_field.cardinality == 'single':
    #                 return ApplyRuleAction,
    #             else:  # optional, multiple
    #                 return ApplyRuleAction, ReduceAction
    #         else:
    #             if hyp.frontier_field.cardinality == 'single':
    #                 return GenTokenAction,
    #             elif hyp.frontier_field.cardinality == 'optional':
    #                 if hyp._value_buffer:
    #                     return GenTokenAction,
    #                 else:
    #                     return GenTokenAction, ReduceAction
    #             else:
    #                 return GenTokenAction, ReduceAction
    #     else:
    #         return ApplyRuleAction,

    # def get_valid_continuating_productions(self, hyp):
    #     if hyp.tree:
    #         if self.grammar.is_composite_type(hyp.frontier_field.type):
    #             return self.grammar[hyp.frontier_field.type]
    #         else:
    #             raise ValueError
    #     else:
    #         return self.grammar[self.grammar.root_type]

 