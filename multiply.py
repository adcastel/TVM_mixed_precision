import math
import tvm
import tvm.testing
from tvm import te
import numpy


def multiply_ints(target):
    A = te.var("A", dtype='int16')
    B = te.var("B", dtype='int16')
    C = tvm.topi.cast(te.compute((1, ), lambda i: te.sum(A * B, axis=0), name='C'), 'int32')
    s = te.create_schedule(C.op)
    
    f = tvm.lower(s, [A, B, C], name="multiply_ints", simple_mode=False)
    func = tvm.build(f, target=target)
    func.save("multiply_ints.s", 's')
    #print(f)
    print(target)
    return func


def multiply_array(target):
    p = te.reduce_axis((0, 1), "p")
    A = te.placeholder((4,), name='A', dtype='int16')
    B = te.placeholder((4,), name='B', dtype='int16')
    C = tvm.topi.cast(te.compute((4,), lambda i: te.sum(A[i] * B[i], axis=p), name='C'), 'int32')
    s = te.create_schedule(C.op)
    x, = C.op.axis
    s[C].vectorize(x)    
    
    f = tvm.lower(s, [A, B, C], name="multiply_array", simple_mode=False)
    func = tvm.build(f, target=target)
    func.save("multiply_array.s", 's')
    #print(f)
    print(target)
    return func


def multiply_array_float(target):
    p = te.reduce_axis((0, 1), "p")
    A = te.placeholder((4,), name='A', dtype='float16')
    B = te.placeholder((4,), name='B', dtype='float16')
    C = tvm.topi.cast(te.compute((4,), lambda i: te.sum(A[i] * B[i], axis=p), name='C'), 'float32')
    s = te.create_schedule(C.op)
    x, = C.op.axis
    s[C].vectorize(x)    
    
    f = tvm.lower(s, [A, B, C], name="multiply_array_float", simple_mode=False)
    func = tvm.build(f, target=target)
    func.save("multiply_array_float16.s", 's')
    #print(f)
    print(target)
    return func


def multiply_add_array(target):
    p = te.reduce_axis((0, 2), "p")
    A = te.placeholder((4,2), name='A', dtype='float32')
    B = te.placeholder((4,2), name='B', dtype='float32')
    C = te.compute((4,2), lambda i, j: te.sum(A[i, j] * B[i, j], axis=p), name='C')
    D_i = te.placeholder((4,2), name='D_i', dtype='float32')
    #D = te.compute((4,2), lambda i, j: tvm.tir.Add(C[i, j], D_i[i, j]), name='D')
    for i in range(8):
        print("C = ", C.value_index)
        print("D_i = ", D_i.value_index)
    D = te.compute((4,2), lambda i, j: C[i, j] + D_i[i, j], name='D')
    s = te.create_schedule(C.op)
    x, y = C.op.axis
    s[C].vectorize(x)    
    
    f = tvm.lower(s, [A, B, C, D_i, D], name="multiply_add_array", simple_mode=False)
    func = tvm.build(f, target=target)
    func.save("multiply_add_array.s", 's')
    #print(f)
    print(target)
    return func
