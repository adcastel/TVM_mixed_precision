import math
import tvm
from tvm import te
import numpy
from multiply import *


def dot(target,dtype_op1,dtype_op2,dtype_store, length=1,unroll=0,vectorize=0,lane = 4):

    p = te.reduce_axis((0, 1), "p")
    if length != 1:
        X = te.placeholder((length,), name='X', dtype=dtype_op1)
        Y = te.placeholder((length,), name='Y', dtype=dtype_op2)
        alpha= te.var("alpha", dtype=dtype_op1)
        
        #Z = te.compute((length,), lambda i: alpha + X[i] * Y[i], name='Z') #, dtype=dtype_store)
        if (dtype_op1 == dtype_op2 and dtype_op1 == dtype_store):
            Z = te.compute((length,), lambda i: te.sum(X[i] * Y[i], axis=p), name='Z') #, dtype=dtype_store)
        else:
            Z = tvm.topi.cast(te.compute((length,), lambda i: te.sum(X[i] * Y[i], axis=p), name='Z') , dtype=dtype_store)

    else:
        X = te.var("X", dtype=dtype_op1)
        Y = te.var("Y", dtype=dtype_op2)
        alpha= te.var("alpha", dtype=dtype_op1)
    
    print('expression dtype: Z({}) = alpha({}) + X({}) * Y({})'.format(Z.dtype, X.dtype, alpha.dtype, Y.dtype))
    s = te.create_schedule(Z.op)
    x,= Z.op.axis
    if vectorize == 1:
        xo,xi = s[Z].split(x, factor=lane)
        s[Z].vectorize(xi)
    n = 'dot_length_{}_op1_{}_op2_{}_opstore_{}'.format(length,dtype_op1,dtype_op2,dtype_store)
    """
     dup     v0.4s, v0.s[0]      #broadcast alpha
     .LBB1_1:
     ldr     q1, [x1, x8]        #load X
     ldr     q2, [x2, x8]        #load Y
     fmla    v2.4s, v1.4s, v0.4s #  Y = Y + X * broadcast(alpha) 
     str     q2, [x0, x8]        #Z = Y
     add     x8, x8, #16
     cmp     x8, #4000
     b.ne    .LBB1_1
    """
    f = tvm.lower(s, [alpha, X, Y, Z], name=n, simple_mode=False)
    print(f)
    rt_mod = tvm.build(f, target="c")#look here
    #print(rt_mod.get_source())
    func = tvm.build(f, target=target)
    func.save(n+".s", 's')
    
    return func


if __name__ == "__main__":
    target = "llvm -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon,+fp16fml"
    dev = tvm.device(target, 0)
    dtype_op1, dtype_op2, dtype_store ="float32","float32", "float32"
    dtype_op1, dtype_op2, dtype_store ="float16","float16", "float16"
    dtype_op1, dtype_op2, dtype_store ="float16","float16", "float32"
    size=1000
    lane = 8
    func = dot(target,dtype_op1, dtype_op2,dtype_store,size,0,1, lane)
    
    if size != 1:
        a = tvm.nd.array(numpy.random.rand(size).astype(dtype_op1), dev)
        b = tvm.nd.array(numpy.random.rand(size).astype(dtype_op2), dev)
        c = tvm.nd.array(numpy.zeros((size), dtype=dtype_store), dev)
        alpha = 3.0
    else:
        a = 200
        b = 200
        cnp = numpy.zeros(1,)
        c = tvm.nd.array(cnp.astype(dtype_store))# tvm.nd.array(numpy.zeros((1,), dtype=dtype_store), dev)
    
    func(alpha, a, b, c)
    print(c)
    """
    # Multiplicación de 2 arrays de enteros 
    func = multiply_array(target="llvm")
    a = tvm.nd.array(numpy.random.randint(1,10,(4, ), 'int16'), dev)
    b = tvm.nd.array(numpy.random.randint(1,10,(4, ), 'int16'), dev)
    c = tvm.nd.array(numpy.random.randint(1,10,(4, ), dtype='int32'), dev)
    func(a, b, c)
    print("C = C + A * B")
    print("A = ", a)
    print("B = ", b)
    print("C = ", c)


    # Multiplicación de 2 arrays de float
    func = multiply_array_float(target=target)  # Xavier
    a = tvm.nd.array(numpy.random.rand(4, ).astype('float16'), dev)
    b = tvm.nd.array(numpy.random.rand(4, ).astype('float16'), dev)
    c = tvm.nd.array(numpy.random.rand(4, ).astype('float32'), dev)
    func(a, b, c)
    print("C = C + A * B")
    print("A = ", a)
    print("B = ", b)
    print("C = ", c)



    # Multiplicación de 2 arrays y suma en resultado

    func = multiply_add_array(target=target)
    a = tvm.nd.array(numpy.random.rand(4, 2).astype('float32'), dev)
    b = tvm.nd.array(numpy.random.rand(4, 2).astype('float32'), dev)
    c = tvm.nd.array(numpy.random.rand(4, 2).astype('float32'), dev)
    d = tvm.nd.array(numpy.random.rand(4, 2).astype('float32'), dev)
    d_init = d
    func(a, b, c, d, d)
    print("C = C + A * B")
    print("D = D + C")
    print("A = ", a)
    print("B = ", b)
    print("C = ", c)
    print("D_init = ", d_init)
    print("D = ", d)

    """
    ## Multiplicación de 2 arrays de f16
    #func = multiply_array_f16(target=target)  # Xavier
    #a = tvm.nd.array(numpy.random.rand(4, ).astype('float16'), dev)
    #b = tvm.nd.array(numpy.random.rand(4, ).astype('float16'), dev)
    #c = tvm.nd.array(numpy.random.rand(4, ).astype('float16'), dev)
    #func(a, b, c)
    #print("C = C + A * B")
    #print("A = ", a)
    #print("B = ", b)
    #print("C = ", c)
