class RegexNode:
    def __init__(self, node_class, children=[], params=[]):
        self.node_class = node_class
        self.children = children
        self.params = params
    
    def logical_form(self):
        """
        Не реализовано
        """
        if len(self.children) + len(self.params) > 0:
            return self.node_class + "(" + ",".join([x.logical_form() for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return self.node_class

    def debug_form(self):
        """
        Не реализовано
        """
        if len(self.children) + len(self.params) > 0:
            return str(self.node_class) + "(" + ",".join([x.debug_form() if x is not None else str(x) for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return str(self.node_class)

    def tokenized_logical_form(self):
        """
        Идейно всё то же. Перебор вариантов нод и рекурсивный вызов.
        """
        toks = []
        if self.node_class in ['orderedquery', 'unorderedquery']:
            toks = ['select']
            children = self.children

            for x in children:
                res = x.tokenized_logical_form()
                toks.extend(res)

            if self.node_class == 'orderedquery':
                #FIXME: Страшный костыль, desc и ?x0 приходят в рандомном порядке, когда будет время, надо разобраться.
                desc = self.params.index('desc')
                other = 0 if desc else 1

                toks.extend(["order", "by", self.params[desc], "(", self.params[other], ")"])

            return toks

        elif self.node_class in ['sum', 'count', 'avg', 'nothing']:
            add_to_res = {"nothing": ''}.get(self.node_class, self.node_class)

            if add_to_res:
                toks.extend([add_to_res, "("])

            dconst_or_conts = self.children[0]

            toks.extend(dconst_or_conts.tokenized_logical_form())

            if add_to_res:
                toks.append(")")
            return toks

        elif self.node_class in ['const', 'distinctconst']:

            distinct = {"distinctconst":'distinct'}.get(self.node_class)

            if distinct:
                toks.append(distinct)

            toks.append(self.params)

            return toks

        else:

            if self.node_class in ['r0','trip']:

                toks = ["where", "{"]
                toks.extend(self.params)
                toks.append('.')

                if self.children[0] is not None:
                    toks.extend(self.children[0].params)
                    toks.append('.')

                toks.append('}')


        return [*filter(lambda x: x, toks)]

def build_regex_ast_from_toks(toks, cur):
    node_class = None
    children = []
    params = []

    if toks[1] not in ["count", "avg", "sum"]:
        node_class = "nothing"
    else:
        cur += 2
        node_class = toks[1]

    if toks[cur+1] == "distinct":
        cur += 2
        params = toks[cur: toks.index('where')][0]
        children.append(RegexNode(node_class, [RegexNode("distinctconst", [None], params)]))
    else:
        cur += 1
        params = toks[cur: toks.index('where')][0]
        children.append(RegexNode(node_class, [RegexNode("const", [None], params)]))

    cur = toks.index('where') + 2

    # print(toks[cur + 4], cur)
    if toks[cur + 4] != '}':
        node_class = "r0"
        chs = toks[cur:cur+3]
        cur += 4

        chs.append(r0(toks, cur))

        children.append(RegexNode(node_class, [chs[3]], [chs[0], chs[1], chs[2]]))
    else:
        node_class = "trip"
        chs = toks[cur:cur + 3]

        cur += 4
        children.append(RegexNode(node_class, [None], [chs[0], chs[1], chs[2]]))
    if "order" in toks:
        node_class = "orderedquery"
        cur = toks.index('order')

        params = []
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
        return RegexNode(next_node_class, [chs[3]], [chs[0], chs[1], chs[2]])
    else:
        next_node_class = "trip1"
        chs = toks[cur:cur + 3]
        cur += 4
        return RegexNode(next_node_class, [None], [chs[0], chs[1], chs[2]])