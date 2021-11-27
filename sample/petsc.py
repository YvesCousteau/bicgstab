from pyamg.util.linalg import norm
import matplotlib
from matplotlib import pyplot as plt
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import time
import numpy as np

def petsc(mat_A,mat_b,mat_x,n,tol,it_max,pc,cond):
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
    ksp.setTolerances(rtol=tol)
    ksp.setIterationNumber(it_max)
    ksp.setConvergenceHistory()
    ksp.getPC().setType(pc)
    # ksp.getPC().setType('none')
    # Début du test
    print('\nTesting BiCGStab')
    # PETSc - Test
    t1=time.time()
    ksp.solve(b_petsc,x_petsc)
    t2=time.time()
    ksp.view()
    # Set value in format for residual calcul
    A = mat_A
    b = mat_b
    x = mat_x
    for i in range(n):
        b[i] = b_petsc.getValue(i)
        x[i] = x_petsc.getValue(i)
        for j in range(n):
            A[i,j] = A_petsc.getValue(i,j)

    residual = np.float128(norm(b - A*x)/norm(b))
    forward_error = np.float128(norm(mat_x - x)/norm(mat_x))
    print('\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0))
    print('residu relative = %g'%(residual))
    print('iter :',ksp.getIterationNumber())
    print('forward error relative :',forward_error)
    print('backward error relative :',residual)
    if (forward_error <= cond*residual):
        print("forward error <= condition * backward error")
    else:
        print("forward error >= condition * backward error")
    plt.semilogy(ksp.getConvergenceHistory(), label="petsc")
    # plt.show()
    # plt.savefig('./data_matrix/cavity{}/petsc_cavity{}.png'.format(path,path))
