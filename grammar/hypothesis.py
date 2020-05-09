from .grammar import *
from .dsl_ast import *
from .transition_system import *
class Hypothesis:
    def __init__(self):
        # unclosed nodes should be tuple of (node, input state)
        # self.unclosed_nodes = []
        # linearized action tree
        self.action_tree = None
        # self.ast = None
        self.score = .0
        self.order = "bfs"
        # self.t = 0
        self.pending_nodes = []
        self.pending_states = []

    def update_pending_node(self):
        # r = self.unclosed_nodes[0]
        # self.unclosed_nodes
        # return r
        if self.order == "dfs":
            self._pending_node_by_dfs(self.action_tree)
        else:    
            self._pending_node_by_bfs(self.action_tree)

    def get_pending_node(self):
        return self.pending_nodes[0], self.pending_states[0]

    def _pending_node_by_dfs(self, node):
        if isinstance(node.action, PlaceHolderAction):
            self.pending_nodes.append(node)
        for x in node.fields:
            self._pending_node_by_dfs(x)

    def _pending_node_by_bfs(self, node):
        search_queue = [node]
        while search_queue:
            cur = search_queue[0]
            if isinstance(cur.action, PlaceHolderAction):
                self.pending_nodes.append(cur)

            search_queue = search_queue[1:]
            search_queue.extend(cur.fields)

    def copy_and_apply_action(self, action, score, updated_states=None):
        new_hyp = self.clone()
        new_hyp.apply_action(action, score, updated_states)
        return new_hyp

    def apply_action(self, action, score, updated_states=None):
        node, _ = self.get_pending_node()
        node.action = action
        self.score = self.score + score
        self.pending_nodes = self.pending_nodes[1:]
        self.pending_states = self.pending_states[1:]
        if isinstance(action, GenTokenAction):
            return 

        elif isinstance(action, ApplyRuleAction):
            assert updated_states is not None
            assert len(action.production.fields) == len(updated_states)
            constructor = action.production.constructor
            new_pending_nodes = [ActionTree(PlaceHolderAction(f.type)) for f in constructor.fields]
            node.fields = new_pending_nodes
            
            if self.order == "dfs":
                self.pending_nodes = new_pending_nodes + self.pending_nodes
                self.pending_states = updated_states + self.pending_states
            else:
                self.pending_nodes = self.pending_nodes + new_pending_nodes
                self.pending_states = self.pending_states + updated_states
        else:
            raise ValueError("Invalid acction type")

    def clone(self):
        hyp = Hypothesis()
        hyp.action_tree = self.action_tree.copy()
        # hyp.ast = 
        hyp.score = self.score
        hyp.order = self.order
        hyp.update_pending_node()
        hyp.pending_states = list(self.pending_states)
        assert len(hyp.pending_nodes) == len(hyp.pending_states)
        return hyp

    @classmethod
    def init_hypothesis(cls, root_type, init_state):
        # node = ActionTree()
        node = ActionTree(PlaceHolderAction(root_type))
        hyp = cls()
        hyp.action_tree = node
        hyp.pending_nodes.append(node)
        hyp.pending_states.append(init_state)
        return hyp

    def is_complete(self):
        return not self.pending_nodes