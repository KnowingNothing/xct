import torch
import tvm
from collections import deque
from error import Warning, INFO, Error, __LINE__

NotSure = 1024

op2node = {}
op2index = {}
index2op = []

class oNode(object):
    '''
    ----------
    operation node:
    ----------
    '''
    def __init__(self, op, total=1):
        self._op = op
        self._shape = []
        for l in op.output(0).shape:
            try:
                tmp = int(l)
            except Exception:
                tmp = NotSure
            self._shape.append(tmp)
        self._feature = []
        for i in range(len(self._shape)):
            self._feature.append(torch.zeros(4, dtype=torch.float32))
        
        self._w = torch.ones(1, requires_grad=True)/total
    
    def get_w(self):
        return self._w
    
    
    def add_axis_feature(self, msg):
        pass

    def collect_feature(self, no, op, msg):
        tmp = torch.zeros(4, dtype=torch.float32)
        for var_name, ab in msg.items():
            tmp += torch.tensor([op2index[op], no, ab['a'], ab['b']], dtype=torch.float32)
        tmp /= len(msg)
        self._feature[no-1] += tmp * op2node[op].get_w()



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
        self._axis_weights = torch.ones(length, requires_grad=True)/length
        self._axis2index = {}
        self._index2axis = []
        self._axis2feature = {}
        counter = 0
        for axis in op.axis:
            self._index2axis.append(axis.var.name)
            self._axis2index[axis.var.name] = counter
            counter += 1

        for axis in op.reduce_axis:
            self._index2axis.append(axis.var.name)
            self._axis2index[axis.var.name] = counter
            counter += 1
    
    def add_axis_feature(self, msg):
        for var_name, feature in msg.items():
            tmp1 = torch.zeros(4, dtype=torch.float32)
            for op_index, val in feature.items():
                node = op2node[index2op[op_index]]
                tmp2 = torch.zeros(4, dtype=torch.float32)
                for v in val:
                    tmp2 = tmp2 + v.get_feature()
                tmp2 /= (len(val))
                tmp1 += tmp2 * node.get_w()
            self._axis2feature[var_name] = tmp1


class VarFeature(object):
    def __init__(self, op_index, index_number, a, b):
        self._feature = torch.tensor([op_index, index_number, a, b], dtype=torch.float32)
    
    def __str__(self):
        return str(self._feature)

    def get_feature(self):
        return self._feature


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

def Visit_Call(call, msg=None):
    old_expr = msg['cur_expr']
    old_vars = msg['vars'] if msg['vars'] else {}
    old_var = msg['cur_var'] if msg['cur_var'] else None
    msg['cur_expr'] = 'call'
    op = call.func
    node = op2node[op]
    for i, expr in enumerate(call.args):        
        msg['vars'] = {"__hidden__":{'a':0, 'b':0}}
        msg['cur_var'] = "__hidden__"
        expr = tvm.ir_pass.Simplify(expr)
        Visit_Expr(expr, msg)
        for var_name, ab in msg['vars'].items():
            feature = VarFeature(op2index[op], i, ab['a'], ab['b'])
            if op2index[op] not in msg['features'][var_name]:
                msg['features'][var_name][op2index[op]] = []
            msg['features'][var_name][op2index[op]].append(feature)
        node.collect_feature(i, msg['origin'], msg['vars'])    

    msg['cur_expr'] = old_expr
    msg['vars'] = old_vars
    msg['cur_Var'] = old_var


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

def Visit_Feature(op):
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

class ComputationGraph(object):
    '''
    -----------------
    computation graph:
    -----------------
    we first build a graph from ops
    then we build nodes from ops
    and use nodes to get features of graph

    '''
    def __init__(self, ops):
        global index2op
        global op2index
        global op2node
        if not isinstance(ops, (list, tuple)):
            ops = [ops]
        self._ends = ops
        self._srcs = []
        self._up_graph = {}
        self._down_graph = {}
        reverse_broad_order = []

        # Restore the graph structure by ops
        q = deque()
        visit = set()
        for op in ops:
            q.append(op)
            visit.add(op)
        while q:
            op = q.popleft()
            reverse_broad_order.append(op)
            if op not in self._up_graph:
                tmp = []
                for t in op.input_tensors:
                    tmp.append(t.op)
                    if t.op not in self._down_graph:
                        self._down_graph[t.op] = []
                    self._down_graph[t.op].append(op)
                    if t.op not in visit:
                        q.append(t.op)
                        visit.add(t.op)
                if not (op.input_tensors):
                    self._srcs.append(op)
                self._up_graph[op] = tmp
            else:
                Warning("Circle in computation graph , mutliple op: {} [line:{}|file:{}]".format(op, __LINE__, __file__))
        
        # First pass
        # build Nodes from ops
        broad_order = list(reversed(reverse_broad_order))
        total_ops = len(broad_order)
        for i, op in enumerate(broad_order):
            if isinstance(op, tvm.tensor.PlaceholderOp):
                op2node[op] = pNode(op, total_ops)
            elif isinstance(op, tvm.tensor.ComputeOp):
                op2node[op] = cNode(op, total_ops)
            # get op2index
            op2index[op] = i
        
        # get index2op
        index2op = broad_order

        # Second pass
        # make relations between nodes
        for op, node in op2node.items():
            features = Visit_Feature(op)
            node.add_axis_feature(features)

                
        


