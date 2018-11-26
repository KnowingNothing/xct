import torch
import torch.nn as nn
import tvm
from collections import deque
from error import Warning, INFO, Error, __LINE__

NotSure = 1024
Node_embed_dim = 8
Axis_feature_dim = 5
Action_embed_dim = 8
Column_dim = 16
Kernel_size_1 = 3


op2node = {}
op2index = {}
index2op = []
src = []
end = []
down_graph = {}
action2index = {}
index2action = []

class oNode(object):
    '''
    ----------
    operation node:
    ----------
    '''
    def __init__(self, op, total=1):
        self._op = op
        self._shape = []
        self._schedule_record = []
        for l in op.output(0).shape:
            try:
                tmp = int(l)
            except Exception:
                tmp = NotSure
            self._shape.append(tmp)
        self._feature = []
        for i in range(len(self._shape)):
            self._feature.append([])    
    
    def add_axis_feature(self, msg):
        pass

    def print_feature(self):
        print("-------------------------------------------------")
        print("op:{} features:".format(self._op))
        print("    output shape: {}".format(self._shape))
        print("    operation feature:")
        for i, f in enumerate(self._feature):
            print("        at dim {}: {}".format(i, f))
        print("-------------------------------------------------")

    def collect_feature(self, where, op, msg):
        '''
        Add a feature about how other operation visits itself
        '''
        tmp = []
        node = op2node[op]
        for var_name, ab in msg.items():
            dom = node.get_dom(var_name)
            no = node.get_pos(var_name)
            tmp.append(torch.tensor([op2index[op], no, ab['a'], ab['b'], dom], dtype=torch.float32))
        self._feature[where].extend(tmp)
    
    def get_dom(self, var_name):
        return 0
    
    def get_pos(self, var_name):
        return -1
    
    def get_feature(self):
        return self._feature



class pNode(oNode):
    '''
    -----------------
    placeholder node:
    -----------------
    '''
    def __init__(self, op, total=1):
        super(pNode, self).__init__(op, total)


class cNode(oNode):
    '''
    -------------
    compute node:
    -------------
    '''
    def __init__(self, op, total=1):
        super(cNode, self).__init__(op, total)
        length = len(op.axis) + len(op.reduce_axis)
        self._axis2index = {}
        self._index2axis = []
        self._axis2feature = {}
        self._axis2dom = {"__hidden__":0}
        counter = 0
        for axis in op.axis:
            self._index2axis.append(axis.var.name)
            try:
                tmp = int(axis.dom.extent)
            except Exception:
                tmp = NotSure
            self._axis2dom[axis.var.name] = tmp
            self._axis2index[axis.var.name] = counter
            counter += 1

        for axis in op.reduce_axis:
            self._index2axis.append(axis.var.name)
            try:
                tmp = int(axis.dom.extent)
            except Exception:
                tmp = NotSure
            self._axis2dom[axis.var.name] = tmp
            self._axis2index[axis.var.name] = counter
            counter += 1
        
        self._axis2index["__hidden__"] = counter
    
    def add_axis_feature(self, msg):
        '''
        add features for an axis of the operation
        '''
        for var_name, feature in msg.items():
            tmp1 = {}
            for op_index, val in feature.items():
                node = op2node[index2op[op_index]]
                tmp2 = []
                for v in val:
                    tmp2.append(v)
                tmp1[op_index] = tmp2
            self._axis2feature[var_name] = tmp1
    
    def get_axis_feature(self, var_name):
        return self._axis2feature[var_name]
        
    def get_dom(self, var_name):
        return self._axis2dom[var_name]
    
    def get_pos(self, var_name):
        return self._axis2index[var_name]
    
    def print_feature(self):
        super(cNode, self).print_feature()
        print("-------------------------------------------------")
        print("    additional feature of cNode:")
        for i, var_name in enumerate(self._index2axis):
            print("    axis {}:".format(var_name))
            for op_index, f in self._axis2feature[var_name].items():
                print("         at op: {}".format(index2op[op_index]))
                for ff in f:
                    print("          {}".format(ff.get_feature()))
        print("-------------------------------------------------")


class VarFeature(object):
    '''
    class containing feature of axis
    '''
    def __init__(self, op_index, index_number, a, b, dom):
        self._feature = torch.tensor([op_index, index_number, a, b, dom], dtype=torch.float32)
    
    def __str__(self):
        return str(self._feature)

    def get_feature(self):
        return self._feature


class Action(object):
    pass

class ComputeAt(Action):
    def __init__(self, consumer, axis):
        super(ComputeAt, self).__init__()
        self._consumer = consumer
        self._axis = axis

class ComputeInline(Action):
    def __init__(self):
        super(ComputeInline, self).__init__()

class Split(Action):
    def __init__(self, axis, nparts):
        super(Split, self).__init__()
        self._axis = axis
        self._nparts = nparts     

class Reorder(Action):
    def __init__(self, axis_list):
        super(Reorder, self).__init__()
        self._axis_list = axis_list

class Fuse(Action):
    def __init__(self, axis_list):
        super(Fuse, self).__init__()
        self._axis_list = axis_list

class Parallel(Action):
    def __init__(self, axis):
        super(Parallel, self).__init__()
        self._axis = axis

class Vectorize(Action):
    def __init__(self, axis):
        super(Parallel, self).__init__()
        self._axis = axis

class Unroll(Action):
    def __init__(self, axis):
        super(Parallel, self).__init__()
        self._axis = axis


'''
Belowing are a series of Visit_XXX functions,
which are used to collect features from the 
graph.

The key factor is the parameter 'msg', which
contains necessary information. To aid understanding,
here are the details about msg:
msg : dict
    {
        'origin_op': Operation <- the Operation for which collect features
        'features' : {
                        var.name : {
                                        op_index : [torch.tensors...] <- features of visiting index2op[op_index] 
                                   }
                     }
        'cur_var' : var.name <- name of var of current scope
        'vars' : {
                    var.name : {
                                    'a': int <- multiply factor
                                    'b': int <- add factor
                               }
                 }
        'cur_exp' : string <- operation type in current scope
    }
And there is always a default var name: __hidden__, 
which is useful in capture const indice information
'''
# Begin of Visits
def Visit_Var(var, msg=None):
    msg['cur_var'] = var.name
    msg['vars'][var.name] = {'a':1, 'b':1}

def Visit_ConstExpr(const_expr, msg=None):
    if msg['cur_expr'] == 'add':
        for var in msg['vars'].keys():
            msg['vars'][var]['b'] = const_expr.value
    if msg['cur_expr'] == 'sub':
        for var in msg['vars'].keys():
            msg['vars'][var]['b'] = -const_expr.value
    elif msg['cur_expr'] == 'mul':
        msg['vars'][msg['cur_var']]['a'] = const_expr.value
    elif msg['cur_expr'] == 'div':
        msg['vars'][msg['cur_var']]['a'] = 1.0 / const_expr.value
    elif msg['cur_expr'] == 'call':
        msg['vars']["__hidden__"]['b'] = const_expr.value

def Visit_Add(add, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'add'
    Visit_Expr(add.a, msg)
    Visit_Expr(add.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Sub(sub, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'sub'
    Visit_Expr(sub.a, msg)
    Visit_Expr(sub.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Mul(mul, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'mul'
    Visit_Expr(mul.a, msg)
    Visit_Expr(mul.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Div(div, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'div'
    Visit_Expr(div.a, msg)
    Visit_Expr(div.b, msg)
    msg['cur_expr'] = old_expr

def Visit_And(_and, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'and'
    Visit_Expr(_and.a, msg)
    Visit_Expr(_and.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Or(_or, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'or'
    Visit_Expr(_or.a, msg)
    Visit_Expr(_or.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Not(_not, msg=None):
    pass

def Visit_CmpExpr(cmp_expr, msg=None):
    old_expr = msg['cur_expr']
    msg['cur_expr'] = 'not'
    Visit_Expr(cmp_expr.a, msg)
    Visit_Expr(cmp_expr.b, msg)
    msg['cur_expr'] = old_expr

def Visit_Reduce(reduce, msg=None):
    for expr in reduce.source:
        Visit_Expr(expr, msg)

def Visit_Cast(cast, msg=None):
    Visit_Expr(cast.value, msg)

def Visit_Select(select, msg=None):
    Visit_Expr(select.true_value, msg)
    Visit_Expr(select.false_value, msg)

# Visit call is special because we care obout it
def Visit_Call(call, msg=None):
    old_expr = msg['cur_expr']
    old_vars = msg['vars'] if msg['vars'] else {}
    old_var = msg['cur_var'] if msg['cur_var'] else None
    msg['cur_expr'] = 'call'
    op = call.func
    node = op2node[op]
    origin_node = op2node[msg['origin']]
    for i, expr in enumerate(call.args):        
        msg['vars'] = {"__hidden__":{'a':0, 'b':0}}
        msg['cur_var'] = "__hidden__"
        expr = tvm.ir_pass.Simplify(expr)
        Visit_Expr(expr, msg)
        for var_name, ab in msg['vars'].items():
            feature = VarFeature(op2index[op], i, ab['a'], ab['b'], origin_node.get_dom(var_name))
            if op2index[op] not in msg['features'][var_name]:
                msg['features'][var_name][op2index[op]] = []
            msg['features'][var_name][op2index[op]].append(feature)
        node.collect_feature(i, msg['origin'], msg['vars'])    

    msg['cur_expr'] = old_expr
    msg['vars'] = old_vars
    msg['cur_Var'] = old_var

# These may be unimportant
def Visit_Let(let, msg=None):
    pass

def Visit_Ramp(ramp, msg=None):
    pass

def Visit_Load(load, msg=None):
    pass

def Visit_Shuffle(shuffle, msg=None):
    pass

def Visit_Broadcast(broadcast, msg=None):
    pass

# Visit Expr dispactches different visits
def Visit_Expr(expr, msg=None):
    p = tvm.expr
    next_steps = {
        p.Var : Visit_Var,
        p.IntImm : Visit_ConstExpr,
        p.UIntImm : Visit_ConstExpr,
        p.FloatImm : Visit_ConstExpr,
        p.StringImm : Visit_ConstExpr,
        p.Add : Visit_Add,
        p.Sub : Visit_Sub,
        p.Mul : Visit_Mul,
        p.Div : Visit_Div,
        p.CmpExpr : Visit_CmpExpr,
        p.And : Visit_And,
        p.Or : Visit_Or,
        p.Not : Visit_Not,
        p.Reduce : Visit_Reduce,
        p.Cast : Visit_Cast,
        p.Select : Visit_Select,
        p.Call : Visit_Call,
        p.Let : Visit_Let,
        p.Ramp : Visit_Ramp,
        p.Load : Visit_Load,
        p.Shuffle : Visit_Shuffle,
        p.Broadcast : Visit_Broadcast
    }
    next_step = next_steps[type(expr)]
    next_step(expr, msg)

# Visit Features starts here
def Visit_Feature(op):
    '''
    Get the feature of an Operation
    -----------
    parameters:
    -----------
    op: Operation
    -----------
    returns:
    -----------
    features: a dict {var.name: {op_index: [torch.tensor...]}}
    '''
    # currently only support computeOp
    if not isinstance(op, tvm.tensor.ComputeOp):
        return {}
    msg = {}
    msg['features'] = {"__hidden__":{}}
    msg['origin'] = op
    for axis in op.axis:
        msg['features'][axis.var.name] = {}
    for axis in op.reduce_axis:
        msg['features'][axis.var.name] = {}
    for body in op.body:
        msg['cur_expr'] = None
        msg['cur_var'] = None
        msg['vars'] = {}
        Visit_Expr(body, msg)
    return msg['features']

# End of Visits

def gennerate_nodes():
    '''
    Generate nodes from ops
    [NOTE] : this will modify op2node, op2index
    -----------
    parameters:
    -----------
    -----------
    returns:
    -----------
    '''
    global op2node
    global op2index
    broad_order = index2op
    total_ops = len(broad_order)
    for i, op in enumerate(broad_order):
        if isinstance(op, tvm.tensor.PlaceholderOp):
            op2node[op] = pNode(op, total_ops)
        elif isinstance(op, tvm.tensor.ComputeOp):
            op2node[op] = cNode(op, total_ops)
        op2index[op] = i

def BFS(ops):
    '''
    Use BFS travel the graph of operations
    [NOTE] : this will modify index2op, src, end, down_graph
    -----------
    parameters:
    -----------
    ops: list of Operation
        same to ops given to tvm.create_schedule
    -----------
    returns:
    -----------
    '''
    global index2op
    global src
    global end
    global down_graph
    reverse_broad_order = []
    srcs = []

    q = deque()
    visit = set()
    for op in ops:
        q.append(op)
        visit.add(op)
    while q:
        op = q.popleft()
        reverse_broad_order.append(op)
        for t in op.input_tensors:
            if t.op not in visit:
                q.append(t.op)
                visit.add(t.op)
            if t.op not in down_graph:
                down_graph[t.op] = []
            down_graph[t.op].append(op)
        if not op.input_tensors:
            srcs.append(op)

    broad_order = list(reversed(reverse_broad_order))
    index2op = broad_order
    end = ops
    src = srcs

def load_actions():
    '''
    Put different actions into a list, and reverse index
    '''
    index2action.extend([ComputeAt, ComputeInline, Split, Reorder, Fuse, Parallel, Vectorize, Unroll])
    for i, action in enumerate(index2action):
        action2index[action] = i

def entry(ops):
    '''
    The entry point
    -----------
    parameters:
    -----------
    ops: list of Operation
        same to ops given to tvm.create_schedule
    -----------
    returns:
    -----------
    '''
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    # Initialize src, end, index2op
    BFS(ops)
    # Initialize op2node, op2index
    gennerate_nodes()
    # Travel the graph and collect features
    for op, node in op2node.items():
        features = Visit_Feature(op)
        node.add_axis_feature(features)
    # prepare actions
    load_actions()


class Agent(nn.Module):
    def __init__(self, actions_len, nodes_len):
        super(Agent, self).__init__()
        # All parameters are here, but not all initialized here
        node2embed = torch.rand([nodes_len, Node_embed_dim], dtype=torch.float32, requires_grad=True)
        action2embed = torch.rand([actions_len, Node_embed_dim], dtype=torch.float32, requires_grad=True)
        weight_node2column = torch.rand([Node_embed_dim, Column_dim], dtype=torch.float32, requires_grad=True)
        weight_axis2column = torch.rand([Axis_feature_dim, Column_dim], dtype=torch.float32, requires_grad=True)
        weight_action2column = torch.rand([Action_embed_dim, Column_dim], dtype=torch.float32, requires_grad=True)
        Kernel_whole_graph = torch.rand([Column_dim * Kernel_size_1, Column_dim], dtype=torch.float32, requires_grad=True)

    def forward(self):
        pass         

    def initialize_node_embed():
    for op in index2op:
        node = op2node[op]
        node2embed[node] = torch.rand(Node_embed_dim, dtype=torch.float32, requires_grad=True)

    def initialize_action_embed():
        
        for action in actions:
            action2embed[action] = torch.rand(Action_embed_dim, dtype=torch.float32, requires_grad=True)

    def get_whole_graph_feature():
        pad_node = torch.zero(Column_dim, dtype=torch.float32)
        padded_graph = []
        for i in range(Kernel_size_1 // 2):
            padded_graph.append(pad_node)
        for i, op in enumerate(index2op):
            padded_graph.append(node2embed[op2node[op]].matmul(weight_node2column))
        for i in range(Kernel_size_1 // 2):
            padded_graph.append(pad_node)
        for op in index2op:
