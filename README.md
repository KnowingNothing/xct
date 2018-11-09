from __future__ import absolute_import, print_function

import tvm
import codecs
import numpy as np

BLOCK_X = int(4)
BLOCK_Y = int(8)
THREAD_X = int(16)
THREAD_Y = int(8)
VTHREAD  = int(2)

num_step =  64 # tvm.var("num_step", dtype="int32")
input_len = int(128)
hidden_len = int(1024)
batch_size = 32 # tvm.var("batch_size", dtype="int32")
gate_num = int(4)

X = tvm.placeholder((num_step, batch_size, input_len), dtype="float32", name="X")
W = tvm.placeholder((gate_num, hidden_len, input_len), dtype="float32", name="W")
H = tvm.placeholder((num_step, batch_size, hidden_len), dtype="float32", name="H")
C = tvm.placeholder((num_step, batch_size, hidden_len), dtype="float32", name="C")
U = tvm.placeholder((gate_num, hidden_len, hidden_len), dtype="float32", name="U")
bias = tvm.placeholder((gate_num, hidden_len), dtype="float32", name="b")

use_X = tvm.placeholder((num_step, batch_size, gate_num, hidden_len), dtype="float32", name="gemm_X")

k1 = tvm.reduce_axis((0, input_len), name="k1")

gemm_X = tvm.compute((num_step, batch_size, gate_num, hidden_len), lambda t, b, g, h: tvm.sum(X[t, b, k1] * W[g, h, k1], axis=k1), name="gemm_X")
init_c, init_h = tvm.compute((1, batch_size, hidden_len), lambda _, b, h: (0.0, 0.0), name="init_c_h")

k2 = tvm.reduce_axis((0, hidden_len), name="k2")

gemm_H = tvm.compute((num_step, batch_size, gate_num, hidden_len), lambda t, b, g, h: tvm.sum(H[t-1, b, k2] * U[g, h, k2], axis=k2), name="gemm_H")
add_result = tvm.compute((num_step, batch_size, gate_num, hidden_len), lambda t, b, g, h: gemm_H[t, b, g, h] + use_X[t, b, g, h] + bias[g, h], name="add_result")
i_gate = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: tvm.sigmoid(add_result[t, b, 0, h]), name="i_gate")
j_trans = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: tvm.tanh(add_result[t, b, 1, h]), name="j_trans")
f_gate = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: tvm.sigmoid(add_result[t, b, 2, h]), name="f_gate")
o_gate = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: tvm.sigmoid(add_result[t, b, 3, h]), name="o_gate")

next_c = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: i_gate[t, b, h] * j_trans[t, b, h] + C[t-1, b, h] * f_gate[t, b, h], name="next_c")
next_h = tvm.compute((num_step, batch_size, hidden_len), lambda t, b, h: o_gate[t, b, h] * tvm.tanh(next_c[t, b, h]), name="next_h")

scan_c, scan_h = tvm.scan([init_c, init_h], [next_c, next_h], [C, H], inputs=[use_X])

s0 = tvm.create_schedule(gemm_X.op)
s = tvm.create_schedule(scan_c.op)

######################################################
# schedule for gemm_X
# best 29ms
#
blockx = tvm.thread_axis("blockIdx.x")
blocky = tvm.thread_axis("blockIdx.y")
blockz = tvm.thread_axis("blockIdx.z")
threadx = tvm.thread_axis((0, THREAD_Y), "threadIdx.x")
thready = tvm.thread_axis((0, THREAD_Y), "threadIdx.y")
vthreadx = tvm.thread_axis((0, VTHREAD), "vthread", name="vx")
vthready = tvm.thread_axis((0, VTHREAD), "vthread", name="vy")


print("check {}".format(s0[gemm_X].op.reduce_axis))
GL = s0.cache_write(gemm_X, "local")
print("check {}".format(s0[gemm_X].op.reduce_axis))
print("check {}".format(s0[GL].op.reduce_axis))
ni, bi, gi, hi = s0[gemm_X].op.axis
s0[gemm_X].bind(ni, blockz)
s0[gemm_X].bind(bi, blocky)
s0[gemm_X].bind(gi, blockx)
hio, hii = s0[gemm_X].split(hi, nparts=THREAD_Y)
s0[gemm_X].bind(hio, thready)
hiio, hiii = s0[gemm_X].split(hii, nparts=THREAD_X)
s0[gemm_X].bind(hiio, threadx)

s0[GL].compute_at(s0[gemm_X], hiio)

ro, ri = s0[GL].split(s0[GL].op.reduce_axis[0], factor=8)

############################################################
# schedule for LSTM
# 
s[add_result].compute_inline()
s[i_gate].compute_inline()
s[j_trans].compute_inline()
s[f_gate].compute_inline()
s[o_gate].compute_inline()
print("check {}".format(s.stages))
print("check {}".format(s[init_c].all_iter_vars))
###########################################################
# schedule init
#
_, bi, hi, = s[init_c].op.axis
bo, bi = s[init_c].split(bi, factor=BLOCK_X)
s[init_c].bind(bi, blockx)
bo, bi = s[init_c].split(bo, factor=BLOCK_Y)
s[init_c].bind(bi, blocky)
s[init_c].bind(bo, blockz)
ho, hi = s[init_c].split(hi, nparts=THREAD_Y)
s[init_c].bind(ho, thready)
ho, hi = s[init_c].split(hi, nparts=THREAD_X)
s[init_c].bind(ho, threadx)

###########################################################
# schedule gemm_H
#
print("check {}".format(s[gemm_H].all_iter_vars))
_, b, g, h = s[gemm_H].op.axis
bo, bi = s[gemm_H].split(b, factor=BLOCK_X)
s[gemm_H].bind(bi, blockx)
bo, bi = s[gemm_H].split(bo, factor=BLOCK_Y)
s[gemm_H].bind(bi, blocky)
s[gemm_H].bind(bo, blockz)
ho, hi = s[gemm_H].split(h, nparts=THREAD_Y)
s[gemm_H].bind(ho, thready)
ho, hi = s[gemm_H].split(hi, nparts=THREAD_X)
s[gemm_H].bind(ho, threadx)

#############################################################
# schedule next_c
#
print("check {}".format(s[next_c].all_iter_vars))
_, bi, hi, = s[next_c].op.axis
bo, bi = s[next_c].split(bi, factor=BLOCK_X)
s[next_c].bind(bi, blockx)
bo, bi = s[next_c].split(bo, factor=BLOCK_Y)
s[next_c].bind(bi, blocky)
s[next_c].bind(bo, blockz)
ho, hi = s[next_c].split(hi, nparts=THREAD_Y)
s[next_c].bind(ho, thready)
ho, hi = s[next_c].split(hi, nparts=THREAD_X)
s[next_c].bind(ho, threadx)

#############################################################
# schedule next_h
#
print("check {}".format(s[next_h].all_iter_vars))
_, bi, hi, = s[next_h].op.axis
bo, bi = s[next_h].split(bi, factor=BLOCK_X)
s[next_h].bind(bi, blockx)
bo, bi = s[next_h].split(bo, factor=BLOCK_Y)
s[next_h].bind(bi, blocky)
s[next_h].bind(bo, blockz)
ho, hi = s[next_h].split(hi, nparts=THREAD_Y)
s[next_h].bind(ho, thready)
ho, hi = s[next_h].split(hi, nparts=THREAD_X)
s[next_h].bind(ho, threadx)

with codecs.open("lstm-gpu-test.cc", "w", encoding="utf-8") as f:
    # f.write(str(tvm.lower(s0, [X, W, gemm_X], simple_mode=True)))
    # f.write(str(tvm.lower(s, [use_X, U, bias, H], simple_mode=True)))
    batch_size_ = 32
    num_step_ = 64
    func = tvm.build(s0, [X, W, gemm_X], "cuda")
    ctx = tvm.gpu(0)
    # x_np = np.random.uniform(-10, 10, size=(num_step_, batch_size_, input_len)).astype(X.dtype)
    # w_np = np.random.uniform(-10, 10, size=(gate_num, hidden_len, input_len)).astype(W.dtype)
    # x = tvm.nd.array(x_np, ctx)
    # w = tvm.nd.array(w_np, ctx)
    # gemm_X_np = np.zeros(shape=(num_step_, batch_size_, gate_num, hidden_len), dtype=np.float32)
    # res = tvm.nd.array(gemm_X_np, ctx)
    # evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
    # print("GEMM  {} ms".format(evaluator(x, w, res).mean * 1e3))
    # f.write(str(func.imported_modules[0].get_source()))

    func = tvm.build(s, [use_X, U, bias, H], "cuda")
    use_x = tvm.nd.array(np.random.uniform(-1,1, size=(num_step_, batch_size_, gate_num, hidden_len)).astype(use_X.dtype), ctx)
    u = tvm.nd.array(np.random.uniform(-1,1,size=(gate_num, hidden_len, hidden_len)).astype(U.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(-1,1,size=(gate_num,hidden_len)).astype(bias.dtype), ctx)
    h = tvm.nd.array(np.zeros(shape=(num_step_, batch_size_, hidden_len), dtype=np.float32), ctx)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
    print("LSTM  {} ms".format(evaluator(use_x, u, b, h).mean * 1e3))
    f.write(str(func.imported_modules[0].get_source()))
