import numpy as np

def conv_backward_naive(x, w, b, dout, pad, stride):
    # N: batch size
    # C: channels
    # H: height
    # W: width
    # _: same as C
    # F: filter number
    # HH: height of kernel
    # WW: width of kernel
    N, C, H, W   = x.shape
    F, _, HH, WW = w.shape
    _,_,H_out,W_out = dout.shape

    x_pad = np.zeros((N, C, H+2*pad, W+2*pad))
    for n in range(N):
        for c in range(C):
            x_pad[n,c] = np.pad(x[n,c],(pad,pad),'constant', constant_values=(0,0))

    db = np.zeros((F))
    dw = np.zeros(w.shape)
    dx_pad = np.zeros(x_pad.shape)

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                current_x_matrix = x_pad[n, :, i * stride: i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    dw[f] = dw[f] + dout[n,f,i,j]* current_x_matrix
                    dx_pad[n,:, i*stride: i*stride+HH, j*stride:j*stride+WW] += w[f]*dout[n,f,i,j]
                db = db + dout[n,:,i,j]
    dx = dx_pad[:,:,pad:H+pad,pad:W+pad]

    return (dw, db, dx)
