
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
import yaml
from yaml import load, dump
import os.path
from os import path


import sample.bicgstab
from sample import bicgstab
import sample.petsc
from sample import petsc

# Main - test
if __name__ == '__main__':
    path_cavity = '06'
    size = '10x10'
    renolds_nomber='200'

    mat_b = mmread('data_matrix/cavity{}/cavity{}_b.mtx'.format(path_cavity,path_cavity))
    mat_x = mmread('data_matrix/cavity{}/cavity{}_x.mtx'.format(path_cavity,path_cavity))
    mat_A = mmread('data_matrix/cavity{}/cavity{}.mtx'.format(path_cavity,path_cavity))
    mat_A = mat_A.toarray()
    n = np.size(mat_b)
    it = 100000
    tol = 1e-15
    plt.title("Convergence History Bi-CGSTAB : Size {} Renolds number {}".format(size,renolds_nomber))
    print('Condition Number =',cond(mat_A))
    print('residual relative (exact) = ',((norm(mat_b - mat_A*mat_x)/norm(mat_b))))
    print('Tolérance relative =',tol)
    print('Itération max =',it)

    residual = str(np.float128(norm(mat_b - mat_A*mat_x)/norm(mat_b)))
    yaml_data = {
        "condition_number":str(cond(mat_A)),
        "matrix A (norm)":str(norm(mat_A)),
        "vector b (norm)":str(norm(mat_b)),
        "lenght b":np.size(mat_b),
        "it max":it,
        "tol":tol,
        "exact":{
            "vector x (norm)":str(norm(mat_x)),
            "residual relative":residual,
        },
    }

    petsc.petsc(mat_A,mat_b,mat_x,n,tol,it,'none',cond(mat_A),yaml_data)
    bicgstab.bcgs(mat_A,mat_b,mat_x,n,tol,it,cond(mat_A),yaml_data)

    if not os.path.isdir('./    docs/cavity_{}'.format(path_cavity)):
        os.makedirs('./docs/cavity_{}'.format(path_cavity))

    file = open('./docs/cavity_{}/yaml_cavity_{}_{}_{}.txt'.format(path_cavity,path_cavity,size,renolds_nomber), "w")
    yaml.dump(yaml_data, file,Dumper=yaml.SafeDumper,sort_keys=False)
    file.close()

    plt.xlabel("itération")
    plt.ylabel("convergence")
    plt.legend()
    plt.savefig('./docs/cavity_{}/figure_cavity_{}_{}_{}.png'.format(path_cavity,path_cavity,size,renolds_nomber))
