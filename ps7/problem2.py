import numpy as np
from potential import sum_neighbors

def apply_laplace(mat,mask):
    tmp=mat.copy()
    tmp[mask]=0
    #since the boundary is now masked, summing over the non-boundary
    #neighbors is now the same as summing over all neighbors.
    tot=sum_neighbors(tmp)
    tot[mask]=0
    return tmp-0.25*tot