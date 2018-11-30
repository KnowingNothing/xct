import torch
import tvm
import Environment as Env
from Model import Agent


class AutoSchedule(object):
    def __init__(self, ops):
        self._ops = ops 
        self._env = Env
        self._env.entry(ops)
        self._agent = Agent(Env.Node_feature_dim, Env.Axis_feature_dim, 32)
        self._agent.reset(Env.src, Env.end, Env.down_graph, Env.op2node, Env.op2index)
        
    def start(self):
        for op in self._env.index2op:
            node = self._env.op2node[op]
            # for placeholder nodes
            if isinstance(node, Env.pNode):
                self._agent.forward(op, self._env.down_graph, self._env.op2node, self._env.op2index, 'none')
                continue
            # for compute nodes
            # first, compute inline
            res = self._agent.forward(op, self._env.down_graph, self._env.op2node, self._env.op2index, 'inline')
            if res[0] > res[1]:
                pass
    
    def try_compute_inline(self, index):
        s = tvm.create_schedule(self._ops)
        for i in range(index + 1):
            op = self._env.index2op[i]
            node = self._env.op2node[op]
            for sch in node.get_schedule_list():
                sch.apply(s, op)
