class RegexNode:
    def __init__(self, node_class, children=[], params=[]):
        self.node_class = node_class
        self.children = children
        self.params = params
    
    def logical_form(self):
        if len(self.children) + len(self.params) > 0:
            return self.node_class + "(" + ",".join([x.logical_form() for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return self.node_class

    def debug_form(self):
        if len(self.children) + len(self.params) > 0:
            return str(self.node_class) + "(" + ",".join([x.debug_form() if x is not None else str(x) for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return str(self.node_class)

    def tokenized_logical_form(self):
        if len(self.children) + len(self.params) > 0:
            toks = [self.node_class] + ["("]
            toks.extend(self.children[0].tokenized_logical_form())
            for c in self.children[1:]:
                toks.append(",")
                toks.extend(c.tokenized_logical_form())
            for p in [str(x) for x in self.params]:
                toks.append(",")
                toks.append(p)
            toks.append(")")
            return toks
        else:
            return [self.node_class]

def build_regex_ast_from_toks(toks, cur):
    node_class = None
    children = []
    params = []

    while cur < len(toks):
        head = toks[cur]
        if head.startswith("<") and head.endswith(">"):
            return RegexNode(head), cur + 1
        elif head == ")":
            return RegexNode(node_class, children, params), cur + 1
        elif head == "(" or head == ",":
            next_tok = toks[cur + 1]
            if next_tok.isdigit():
                params.append(int(next_tok))
                cur = cur + 2
            elif head == "(" and next_tok == ")":
                return RegexNode(node_class), cur + 2
            else:
                ret_vals = build_regex_ast_from_toks(toks, cur + 1)
                children.append(ret_vals[0])
                cur = ret_vals[1]
        else:
            node_class = head
            cur = cur + 1
    print(cur, len(toks))
    print(toks)
    raise ValueError("unsuccessfully parseds ast")