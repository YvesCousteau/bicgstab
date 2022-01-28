
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
import shutil
import sample.bicgstab
from sample import bicgstab
import sample.petsc
from sample import petsc
from numpy import savetxt

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print('My rank is ',rank)
# Main - test


if __name__ == '__main__':
    path = ''
    name='circuit_1'
    mat_b = mmread('data_matrix/{}{}/{}{}_b.mtx'.format(name,path,name,path))
    mat_x = mmread('data_matrix/{}{}/{}{}_x.mtx'.format(name,path,name,path))
    mat_A = mmread('data_matrix/{}{}/{}{}.mtx'.format(name,path,name,path))
    mat_A = mat_A.toarray()
    # savetxt('data_matrix/mat/{}_A.csv'.format(name), mat_A, delimiter=' ')
    # savetxt('data_matrix/mat/{}_b.csv'.format(name), mat_b, delimiter=' ')
    n = np.size(mat_b)
    it_max = n
    tol = 1e-12

    print('Condition Number =',cond(mat_A))
    print('residual relative (exact) = ',((norm(mat_b - mat_A*mat_x)/norm(mat_b))))
    print('Tolérance relative =',tol)
    print('Itération max =',it_max)


    residual = str(np.float128(norm(mat_b - mat_A*mat_x)/norm(mat_b)))
    yaml_data = {
        "condition_number":str(cond(mat_A)),
        "matrix A (norm)":str(norm(mat_A)),
        "vector b (norm)":str(norm(mat_b)),
        "lenght b":np.size(mat_b),
        "it max":it_max,
        "tol":tol,
        "exact":{
            "vector x (norm)":str(norm(mat_x)),
            "residual relative":residual,
        },
    }
    print("zizi")
    petsc.petsc(mat_A,mat_b,mat_x,n,tol,it_max,'none',cond(mat_A),yaml_data)
    bicgstab.bcgs(mat_A,mat_b,mat_x,n,tol,it_max,cond(mat_A),yaml_data)

    if not (os.path.isdir('./docs/{}{}'.format(name,path))):
        os.makedirs('./docs/{}{}'.format(name,path))
    else:
        shutil.rmtree('./docs/{}{}'.format(name,path))
        os.makedirs('./docs/{}{}'.format(name,path))


    file = open('./docs/{}{}/yaml_{}{}_{}.txt'.format(name,path,name,path,n), "w")
    yaml.dump(yaml_data, file,Dumper=yaml.SafeDumper,sort_keys=False)
    file.close()


    plt.title("Convergence History Bi-CGSTAB : Size {}".format(n))
    plt.xlabel("itération")
    plt.ylabel("convergence")
    plt.legend()
    plt.savefig('./docs/{}{}/plt_{}{}_{}.png'.format(name,path,name,path,n))
