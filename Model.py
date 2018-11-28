import torch
import torch.nn as nn
from torch.nn import Parameter as P
from error import INFO, Warning, Error, __LINE__



class NodeCell(nn.Module):
    def __init__(self, feature_dim, hidden):
        super(NodeCell, self).__init__()
        self._hidden = hidden
        self._W = nn.Linear(feature_dim + hidden, hidden)
        self._U = nn.Linear(feature_dim + hidden, hidden)
        self._trans = nn.Linear(feature_dim + hidden, hidden)

    def forward(self, x, children_hs):
        num = len(children_hs)
        h = torch.sum(torch.stack(children_hs), dim=0) / num
        r = torch.sigmoid(self._W(torch.cat([x, h])))
        z = torch.sigmoid(self._U(torch.cat([x, h])))
        trans = [torch.tanh(self._trans(torch.cat([x, r*h_]))) for h_ in children_hs]
        next_h = torch.sum(torch.stack([(1 - z) * children_hs[i] + z * trans[i] for i in range(num)]), dim=0) / num
        return next_h


class AxisCell(nn.Module):
    def __init__(self, feature_dim, hidden):
        super(AxisCell, self).__init__()
        self._hidden = hidden
        self._Linear = nn.Linear(feature_dim + hidden, hidden * 2)
        self._trans = nn.Linear(feature_dim + hidden, hidden)
        
    def forward(self, x, h):
        pos1 = self._hidden
        activated = torch.sigmoid(self._Linear(torch.cat([x, h])))
        r = activated[:pos1]
        z = activated[pos1:]
        trans = torch.tanh(self._trans(torch.cat([x, r*h])))
        next_h = (1 - z) * h + z * trans
        return next_h


class Agent(nn.Module):
    def __init__(self, Node_feature_dim, Axis_feature_dim, hidden, num_actions, actions_embed=None, freeze_embed=False):
        super(Agent, self).__init__()
        # All parameters are here, but not all initialized here
        if actions_embed is None:
            actions_embed = torch.eye(num_actions)
        self._hidden = hidden
        self._action2embed = nn.Embedding.from_pretrained(actions_embed, freeze_embed)
        self._action_cell = AxisCell(self._action2embed.weight.shape[1], hidden)
        self._schedule_cell = NodeCell(hidden * 2, hidden)
        self._node_cell = NodeCell(Node_feature_dim, hidden)
        self._axis_cell = AxisCell(Axis_feature_dim, hidden)
        self._reduce_axis_cell = AxisCell(Axis_feature_dim, hidden)
        self._attention = P(torch.rand([2 * hidden, hidden], requires_grad=True))
        self._Output1 = nn.Linear(hidden * 3, hidden)
        self._Output2 = nn.Linear(hidden, num_actions)
        self._cache_schedule = {}
        self._cache_up_states = {}
        self._cache_down_states = {}
        self._last_h = torch.zeros(hidden)

        self._new = False
        self._first_call = True

    def forward(self, op, op2node):
        if self._first_call and not self._new:
            Warning("You forget to reset model before using")
        self._first_call = False
        attention_state = self._get_attention(self._last_h)
        region_state = self._get_region_feature(op, op2node)
        state = torch.cat([attention_state, region_state])
        tmp = self._Output1(state)
        self._last_h = tmp
        res = self._Output2(tmp)
        return torch.softmax(res, dim=0)

    def reset(self, src, end, down_graph, op2node, op2index):
        self._cache_schedule = {}
        self._cache_up_states = {}
        self._cache_down_states = {}
        self._last_h = torch.zeros(self._hidden)
        self._get_whole_graph_feature(src, end, down_graph, op2node, op2index)
        self._new = True

    def _get_attention(self, h):
        cache_attentions = {}
        total = torch.tensor(0.0)
        for op, up_state in self._cache_up_states.items():
            down_state = self._cache_down_states[op]
            state = torch.cat([down_state, up_state])
            cache_attentions[op] = state.matmul(self._attention).matmul(h)
            total += cache_attentions[op]
        attention_state = torch.zeros(self._hidden * 2)
        for op, val in cache_attentions.items():
            tmp = val / total
            state = torch.cat([self._cache_down_states[op], self._cache_up_states[op]])
            attention_state += tmp * state 
        return attention_state

    def _get_down_graph_feature(self, op, op2node, op2index):
        if op is None:
            return
        node = op2node[op]
        index = op2index[op]
        if not op.input_tensors:
            self._cache_down_states[op] = self._node_cell(node.get_feature(), [torch.zeros(self._hidden)])
            return
        axis_cache = []
        for p in op.input_tensors:
            p_op = p.op
            p_node = op2node[p_op]
            self._get_down_graph_feature(p_op, op2node, op2index)
            axis_count = len(op.axis) if hasattr(op, 'axis') else 0
            features = p_node.get_use_feature()
            for i in range(axis_count):
                for feature in features[i][index]:
                    # feature[1] indicates which axis in op visits p_op
                    if feature[1] >= axis_count:
                        axis_cache.append(self._reduce_axis_cell(feature, self._cache_down_states[p_op]))
                    else:
                        axis_cache.append(self._axis_cell(feature, self._cache_down_states[p_op]))
        self._cache_down_states[op] = self._node_cell(node.get_feature(), axis_cache)
    
    def _get_up_graph_feature(self, op, down_graph, op2node, op2index):
        if op is None:
            return
        node = op2node[op]
        if op not in down_graph:
            self._cache_up_states[op] = self._node_cell(node.get_feature(), [torch.zeros(self._hidden)])
            return
        axis_cache = []
        for c_op in down_graph[op]:
            c_node = op2node[c_op]
            c_index = op2index[c_op]
            self._get_up_graph_feature(c_op, down_graph, op2node, op2index)
            axis_count = len(c_op.axis) if hasattr(c_op, 'axis') else 0
            features = node.get_use_feature()
            for i in range(axis_count):
                for feature in features[i][c_index]:
                    if feature[1] >= axis_count:
                        axis_cache.append(self._reduce_axis_cell(feature, self._cache_up_states[c_op]))
                    else:
                        axis_cache.append(self._axis_cell(feature, self._cache_up_states[c_op]))
        self._cache_up_states[op] = self._node_cell(node.get_feature(), axis_cache)

    def _get_whole_graph_feature(self, src, end, down_graph, op2node, op2index):
        for op in end:
            self._get_down_graph_feature(op, op2node, op2index)
        for op in src:
            self._get_up_graph_feature(op, down_graph, op2node, op2index)
    
    def _get_region_feature(self, op, op2node):
        p_cache = []
        for p in op.input_tensors:
            p_op = p.op
            if p_op in self._cache_schedule:
                p_cache.append(self._cache_schedule[p_op])
                continue
            p_node = op2node[p_op]
            schedule_list = p_node.get_schedule_list()
            act_embed_input= []
            for act_no, act in schedule_list:
                act_embed_input.append(act_no)
            act_embed_input = torch.LongTensor(act_embed_input)
            act_embed = self._action2embed(act_embed_input)
            h = self._cache_down_states[p_op]
            for i in range(len(schedule_list)):
                h = self._action_cell(act_embed[i], h)
            self._cache_schedule[p_op] = h
            p_cache.append(h)
        if not p_cache:
            p_cache.append(torch.zeros(self._hidden))
        p_state = self._schedule_cell(torch.cat([self._cache_down_states[op], self._cache_up_states[op]]), p_cache)
        return p_state





