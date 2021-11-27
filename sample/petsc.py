from pyamg.util.linalg import norm
import matplotlib
from matplotlib import pyplot as plt
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import time
import numpy as np

def petsc(mat_A,mat_b,mat_x,n,tol,it_max,pc,cond,yaml_data):
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
        x_petsc.setValue(i,1)
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

    residual = str(np.float128(norm(mat_b - mat_A*x_petsc)/norm(mat_b)))
    forward_error = str(np.float128(norm(mat_x - x_petsc)/norm(mat_x)))
    cond_backward_error = str(np.float128(cond*np.float128((norm(mat_b - mat_A*x_petsc)/norm(mat_b)))))
    yaml_data["petsc"] = {
        "solving time (s)":(t2-t1)*1000.0,
        'iter':ksp.getIterationNumber(),
        "vector x (norm)":str(norm(x_petsc)),
        'residual relative':residual,
        "forward error":forward_error,
        "condition * backward error":cond_backward_error,
    }
    # if (forward_error <= cond*residual):
    #     print("forward error <= condition * backward error :",cond*residual)
    # else:
    #     print("forward error >= condition * backward error :",cond*residual)
    plt.semilogy(ksp.getConvergenceHistory(), label="petsc")
    # plt.show()
    # plt.savefig('./data_matrix/cavity{}/petsc_cavity{}.png'.format(path,path))
