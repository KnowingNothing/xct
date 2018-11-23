from error import Warning, INFO, Error, __LINE__
import collections import deque

class ComputationGraph(object):
    def __init__(self, ops):
        self._ends = ops
        self._srcs = []
        self._up_graph = {}
        self._down_graph = {}
        self._broad_order = []
        q = deque()
        visit = set()
        for op in ops:
            q.append(op)
            visit.add(op)
        while q:
            op = q.popleft()
            self._broad_order.append(op)
            if op not in self._up_graph:
                tmp = []
                for t in op.input_tensors:
                    tmp.append(t.op)
                    if t.op not in self._down_graph:
                        self._down_graph[t.op] = []
                    else:
                        self._down_graph[t.op].append(op)
                    if t.op not in visit:
                        q.append(t.op)
                        visit.add(t.op)
                if not (op.input_tensors):
                    self._srcs.append(op)
                self._up_graph[op] = tmp
            else:
                Warning("Circle in computation graph , mutliple op: {} [line:{}|file:{}]".format(op, __LINE__, __file__))
        
        
        



