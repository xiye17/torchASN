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
    children = []
    params = []

    if toks[1] not in ["count", "avg", "sum"]:
        next_node_class = "nothing"
    else:
        cur += 2
        next_node_class = toks[1]

    if toks[cur+1] == "distinct":
        children.append(RegexNode(next_node_class, RegexNode("distinctconst", None, toks[cur+2])))
    else:
        children.append(RegexNode(next_node_class, RegexNode("const", None, toks[cur + 1])))

    cur = toks.index('where') + 2

    # print(toks, cur)
    if toks[cur + 4] != '}':
        next_node_class = "r0"
        chs = toks[cur:cur + 3]
        cur += 4
        chs.append(r0(toks, cur))
        children.append(RegexNode(next_node_class, chs[3], [chs[0], chs[1], chs[2]]))
    else:
        next_node_class = "trip"
        chs = toks[cur:cur + 3]
        cur += 4
        children.append(RegexNode(next_node_class, None, [chs[0], chs[1], chs[2]]))

    if "order" in toks:
        node_class = "orderedquery"
        cur = toks.index('order')
        params.append(toks[cur+2])
        params.append(toks[cur+4])
    else:
        node_class = "unorderedquery"
    return RegexNode(node_class, children, params)


def r0(toks, cur):
    # print(toks, cur)
    if toks[cur + 4] != '}':
        next_node_class = "r1"
        chs = toks[cur:cur + 3]
        cur += 4
        chs.append(r0(toks, cur))
        return RegexNode(next_node_class, chs[3], [chs[0], chs[1], chs[2]])
    else:
        next_node_class = "trip1"
        chs = toks[cur:cur + 3]
        cur += 4
        # print([chs[0], chs[2], chs[1]])
        return RegexNode(next_node_class, None, [chs[0], chs[1], chs[2]])