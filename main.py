
import numpy as np
from numpy import dot
from numpy.linalg import cond
from numpy.linalg import norm
from mpi4py import MPI
import sys
import scipy.io
from scipy.io import mmread
import matplotlib
from matplotlib import pyplot as plt



import sample.bicgstab
from sample import bicgstab
import sample.petsc
from sample import petsc

# Main - test
if __name__ == '__main__':
    path = '06'
    size = '10x10'
    renolds_nomber='200'

    mat_b = mmread('data_matrix/cavity{}/cavity{}_b.mtx'.format(path,path))
    mat_x = mmread('data_matrix/cavity{}/cavity{}_x.mtx'.format(path,path))
    mat_A = mmread('data_matrix/cavity{}/cavity{}.mtx'.format(path,path))
    mat_A = mat_A.toarray()
    n = mat_b.size
    it = 100000
    tol = 1e-15
    plt.title("Convergence History Bi-CGSTAB : Size {} Renolds number {}".format(size,renolds_nomber))
    print('Condition Number =',cond(mat_A))
    print('residual relative (exact) = %g'%(np.float128(norm(mat_b - mat_A*mat_x)/norm(mat_b))))
    print('Tolérance relative =',tol)
    print('Itération max =',it)
    petsc.petsc(mat_A,mat_b,mat_x,n,tol,it,'none',cond(mat_A))
    bicgstab.bcgs(mat_A,mat_b,mat_x,n,tol,it,cond(mat_A))
    plt.xlabel("itération")
    plt.ylabel("convergence")
    plt.legend()
    plt.savefig('./figures/figure_cavity_{}_{}_{}.png'.format(path,size,renolds_nomber))
