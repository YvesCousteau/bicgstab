import numpy as np
import time
import sys
import petsc4py
petsc4py.init(sys.argv)
from numpy import dot
from pyamg.util.linalg import norm, cond
from mpi4py import MPI
from petsc4py import PETSc

import sample.bicgstab
from sample import bicgstab

def normal_distribution(n,tol_init,it_init):
    """

    """
    # Self - Initialisations
    # Création de la matrice de systèmes linéaires par méthode random.normal() pour obtenir une "Normal (Gaussian) Distribution".
    P = np.random.normal(size=[n, n])
    # Le produit d'une matrice par sa transposée est toujours une matrice symétrique
    A = np.dot(P.T, P)
    # Créations et Initialisations des vecteurs b et x
    b = np.ones(n)
    x = np.ones(n)

    # PETSc - Initialisations
    # Création du "Right Hand Solution" vecteur b.
    b_linalg = PETSc.Vec().createSeq(n)
    # Création de la solution du vecteur x.
    x_linalg = PETSc.Vec().createSeq(n)
    # Initialisations des vecteurs b et x
    for i in range(n):
        b_linalg[i] = 1
        x_linalg[i] = 1
    # Initialisation de la Matrice de systèmes linéaires à résoudre
    A_linalg = PETSc.Mat().create(PETSc.COMM_WORLD)
    A_linalg.setSizes([n,n])
    A_linalg.setType('aij')
    A_linalg.setPreallocationNNZ(n)
    for i in range(n):
        for j in range(n):
            A_linalg[i,j] = A[i,j]
    A_linalg.assemble()

    # Initialisation du ksp solver.
    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.setOperators(A_linalg)
    ksp.setFromOptions()
    ksp.setTolerances(tol_init)
    ksp.setIterationNumber(it_init)
    ksp.view()
    ksp.setConvergenceHistory()

    # conditionnement de la Matrice de systèmes linéaires à résoudre
    print('conditionnement =',cond(A))

    # Début du test
    print('\nTesting BiCGStab')

    # Self - Test
    t1=time.time()
    (x,flag,r,iter,conv_history) = bicgstab.bicgstab_solver(A=A,b=b,tol=tol_init,x=x,it_max=it_init)
    t2=time.time()
    print('\n%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%r )
    print('iter :',iter)
    print(conv_history)

    # PETSc - Test
    t1=time.time()
    ksp.solve(b_linalg, x_linalg)
    t2=time.time()
    print('\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%(norm(b_linalg - A_linalg*x_linalg)/norm(b_linalg)) )
    print('iter :',ksp.getIterationNumber())
    print(ksp.getConvergenceHistory())
    # print(ksp.getConvergedReason())


# Main - test
if __name__ == '__main__':
    # Initialisation de la taille de notre matrice
    n = 10
    tol_init = 1e-30
    it_init = 10000
    normal_distribution(n,tol_init,it_init)
