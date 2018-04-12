import numpy as np

def matmul_forward_naive(w, x):
    out = np.matmul(w, x)
    return out

def matmul_backward_naive(dout, w, x):
    dw = np.matmul(dout, x.T)
    dx = np.matmul(w.T, dout)

    return (dw, dx)
