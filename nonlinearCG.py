import numpy as np;

def nonlinearCG(J, Dx, x0, maxIteration):
    x = x0;
    delta = -Dx(x);
    return;
