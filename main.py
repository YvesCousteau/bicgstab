
import numpy as np
import time
import sys
import petsc4py
petsc4py.init(sys.argv)
from numpy import dot
from numpy.linalg import cond
from numpy.linalg import norm
from mpi4py import MPI
from petsc4py import PETSc
import scipy.io
from scipy.io import mmread
import matplotlib
from matplotlib import pyplot as plt

import sample.bicgstab
from sample import bicgstab

def bcgs(mat_A,mat_b,mat_x,n,tol,it_max):
    # Self - Initialisations
    # Le produit d'une matrice par sa transposée est toujours une matrice symétrique
    A = mat_A
    for i in range(n):
        for j in range(n):
            A[i,j] = mat_A[i,j]
    # Création du "Right Hand Solution" vecteur b.
    b = mat_b
    # Création de la solution du vecteur x.
    x = mat_b
    # Initialisations des vecteurs b et x
    for i in range(n):
        b[i] = mat_b[i]
        x[i] = 1
    # Début du test
    print('\nTesting BiCGStab')
    # Self - Test
    t1=time.time()
    (x,flag,r,iter,conv_history) = bicgstab.bicgstab_solver(A=A,b=b,tol=tol,x=x,it_max=it_max)
    t2=time.time()
    print('\n%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%(norm(b - A*x)/norm(b)) )
    print('iter :',iter)
    plt.semilogy(conv_history)
    plt.show()


def petsc(mat_A,mat_b,mat_x,n,tol,it_max):
    # PETSc - Initialisations
    # Initialisation de la Matrice de systèmes linéaires à résoudre
    A_petsc = PETSc.Mat().createAIJ([n, n],nnz=n*n)
    for i in range(n):
        for j in range(n):
            A_petsc.setValue(i,j,mat_A[i,j])
    A_petsc.assemble()
    # Création du "Right Hand Solution" vecteur b.
    b_petsc = A_petsc.createVecLeft()
    # Création de la solution du vecteur x.
    x_petsc = A_petsc.createVecRight()
    # Initialisations des vecteurs b et x
    for i in range(n):
        b_petsc.setValue(i,mat_b[i])
    # Initialisation du ksp solver.
    ksp = PETSc.KSP().create()
    ksp.setType('bcgs')
    ksp.setOperators(A_petsc)
    # ksp.setFromOptions()
    ksp.setTolerances(rtol=tol)
    ksp.setIterationNumber(it_max)
    ksp.setConvergenceHistory()
    ksp.getPC().setType('none')
    # Début du test
    print('\nTesting BiCGStab')
    # PETSc - Test
    # ksp.setUp()
    t1=time.time()
    ksp.solve(b_petsc,x_petsc)
    t2=time.time()
    ksp.view()
    residual = A_petsc * x_petsc - b_petsc
    print('\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%(residual.norm()/b_petsc.norm()))
    print('iter :',ksp.getIterationNumber())

    plt.semilogy(ksp.getConvergenceHistory())
    plt.show()
# Main - test
if __name__ == '__main__':
    path = '06'

    mat_b = mmread('data_matrix/cavity{}/cavity{}_b.mtx'.format(path,path))
    mat_x = mmread('data_matrix/cavity{}/cavity{}_x.mtx'.format(path,path))
    mat_A = mmread('data_matrix/cavity{}/cavity{}.mtx'.format(path,path))
    mat_A = mat_A.toarray()
    n = mat_b.size
    it = 100000
    tol = 1e-12
    print('Condition Number =',cond(mat_A))
    print('residu absolu (exact) = %g'%(norm(mat_b - mat_A*mat_x)/norm(mat_b)) )
    # print('Minimum Singular Value =',mat_b.min())
    # print('Matrix Norm =',norm(mat_A))
    petsc(mat_A,mat_b,mat_x,n,tol,it)
    bcgs(mat_A,mat_b,mat_x,n,tol,it)
