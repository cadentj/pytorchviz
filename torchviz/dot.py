from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable
from functools import partial

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"
# TODO: Turn these into regular expressions
IGNORE = set([    
    "AccumulateGrad",
    "ViewBackward0",
    "SliceBackward0",
    "PermuteBackward0",
    "TBackward0"
])

def get_fn_name(fn):
    return str(type(fn).__name__)

def make_dot(var, params=None, show_saved=False):
    """ Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='10',
                    ranksep='0.1',
                    height='0.2',
                    fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))


    def add_nodes(fn, previous):
        assert not torch.is_tensor(fn)
        if (fn, previous) in seen:
            print("seen: ", get_fn_name(fn))
            return
        seen.add((fn, previous))

        names = []

        for u in fn.next_functions:
            if hasattr(u[0], 'variable'):
                names.append(get_var_name(u[0].variable))
            else: 
                if u[0] is not None:
                    add_nodes(u[0], fn)

        if names:
            name = "\n".join(names)
            dot.node(str(id(fn)), name, fillcolor='lightblue')
        else:
            dot.node(str(id(fn)), get_fn_name(fn))

        dot.edge(str(id(fn)), str(id(previous)))   

    def add_nodes(fn, start_fn, last_fn):
        

    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn, None)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view(): # add node in place for views
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
