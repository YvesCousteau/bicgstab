"""
Implémentation en Python de l'algorithme de stabilisation du gradient bi-conjugué (Bi-CGSTAB).
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.bicgstab.html

"""
import numpy as np
import time
import sys
import petsc4py
petsc4py.init(sys.argv)
from numpy import array, inner, conjugate, ravel, diag
from pyamg.util.linalg import norm
from scipy.sparse.linalg.isolve.utils import make_system
from mpi4py import MPI
from petsc4py import PETSc

class BiCGSTAB:
    def __init__(self,**kwargs):
        pass

    def solve(A, b, x_init=None, it_max=None, tol=1e-5, prec_A=None, callback=None, residuals=None, xtype=None):
        """
        Résolution de systèmes linéaires non symétriques avec la résolution par la droite (right_hand => vector numpy array).
        =======================
        Algorithme : BICGSTAB
        =======================
        Input :
            A : {array, matrix, sparse matrix, LinearOperator}
                Matrice de systèmes linéaires à résoudre
            b : {array, matrix}
                "Right Hand" du système linéaire
            x_init : {array, matrix}
                Supposition de départ pour la solution (initialement un vecteur de zéro)
            it_max : int
                Maximum d'itérations (initialement a None)
            tol : float
                Tolérance (initialement a 1e-5)
            prec_A : {array, matrix, sparse matrix, LinearOperator}
                Preconditioneur de A (initialement a None)
            callback : function
                La fonction fournie dans le callback est appelée après chaque itération (initialement a None)
            residuals : list
                Historique des normes résiduelles
            xtype : dtype
                dtype de la solution, par défaut la détection automatique du type.
        Output :
            x_final : Solution approchée
            info : statut d'arrêt du bicgstab
        -----------------------
        References
        .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
        Second Edition", SIAM, pp. 231-234, 2003
        http://www-users.cs.umn.edu/~saad/books.html

        =======================
        """

        # Convert inputs to linear system
        (A, prec_A, x, b, postprocess) = make_system(A, prec_A, x_init, b)

        # Check iteration numbers
        if it_max == None:
            it_max = len(x) + 5
        elif it_max < 1:
            raise ValueError('Number of iterations must be positive')

        # Prep for method
        r = b - A*x

        # Calc the norm of r
        normr = norm(r)

        # Set initial residuals if residuals is empty
        if residuals is not None:
            residuals[:] = [normr]

        # Check initial guess ( scaling by b, if b != 0, must account for case when norm(b) is very small)
        normb = norm(b)
        if normb == 0.0:
            normb = 1.0

        # Stopping condition
        if normr < tol*normb:
            return ((x_init),0)

        # Raise  tol if norm r isn't small
        if normr != 0.0:
            tol = tol*normr

        # Check if matrix A is on 1 dimension
        # ravel => A 1-D array, containing the elements of the input, is returned.
        if A.shape[0] == 1:
            entry = ravel(A*array([1.0], dtype=xtype))
            return ((b/entry), 0)

        # simple copy of r
        rstar = r.copy()
        p = r.copy()

        # Ordinary inner product of vectors for 1-D arrays (without complex conjugation)
        # The complex conjugate of a complex number is obtained by changing the sign of its imaginary part
        delta = inner(rstar.conjugate(), r)

        # Set iteration
        iter = 0

        # Loop
        while True:

            # Init preconditioner
            if prec_A is not None:
                A_matrix_p = A*prec_A*p
                matrix_p = prec_A*p
                pass
            else :
                A_matrix_p = A*p
                matrix_p = p
                pass

            # alpha = (r_{iter}, rstar) / (A*p_{iter}, rstar)
            alpha = delta/inner(rstar.conjugate(), A_matrix_p)

            # s_{iter} = r_{iter} - alpha*A*p_{iter}
            s   = r - alpha*A_matrix_p

            # Init preconditioner
            if prec_A is not None:
                A_matrix_s = A*prec_A*s
                matrix_s = prec_A*s
                pass
            else :
                A_matrix_s = A*s
                matrix_s = s
                pass

            # omega = (A*s_{iter}, s_{iter})/(A*s_{iter}, A*s_{iter})
            omega = inner(A_matrix_s.conjugate(), s)/inner(A_matrix_s.conjugate(), A_matrix_s)

            # x_{iter+1} = x_{iter} +  alpha*p_{iter} + omega*s_{iter}
            x = x + alpha*matrix_p + omega*matrix_s

            # r_{iter+1} = s_{iter} - omega*A*s
            r = s - omega*A_matrix_s

            # beta_{iter} = (r_{iter+1}, rstar)/(r_{iter}, rstar) * (alpha/omega)
            delta_new = inner(rstar.conjugate(), r)
            beta = (delta_new / delta) * (alpha / omega)
            delta = delta_new

            # p_{iter+1} = r_{iter+1} + beta*(p_{iter} - omega*A*p)
            p = r + beta*(p - omega*A_matrix_p)

            iter += 1

            normr = norm(r)

            # Stopping conditions
            if residuals is not None:
                residuals.append(normr)

            if callback is not None:
                callback(x)

            if normr < tol:
                return ((x), 0)

            if iter == it_max:
                return ((x), iter)

            pass
        pass

if __name__ == '__main__':
    from pyamg.gallery import poisson

    A = poisson((10,10))
    b = np.ones((A.shape[0],))

    print('\nTesting BiCGStab with %d x %d 2D Laplace Matrix\n'%(A.shape[0],A.shape[0]))

    # Hugo
    t1=time.time()
    (x,flag) = BiCGSTAB.solve(A=A,b=b,x_init=None,tol=1e-8,it_max=100)
    t2=time.time()
    print('%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0))
    print('norm = %g'%(norm(b - A*x)))
    print('info flag = %d'%(flag))

    # PETSc
    n = 32
    # grid spacing
    h = 1.0/(n+1)
    A = PETSc.Mat().create()
    A.setSizes([n**2, n**2])
    A.setType('python')
    # shell = Del2Mat(n) # shell context
    # A.setPythonContext(shell)
    A.setUp()

    x_ksp = np.ones((A.shape[0],))

    ksp = PETSc.KSP().create()
    ksp.setType('bcgs')

    prec = ksp.getPC()
    prec.setType('none')

    ksp.setOperators(A)
    ksp.setFromOptions()

    t1=time.time()
    ksp.solve(b, x_ksp)
    t2=time.time()
    print('%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0))
    print('norm = %g'%(norm(b - A*x)))

    # t1=time.time()
    # (y,flag) = bicgstab_PETSc(A=A,b=b,tol=1e-8,maxiter=100)
    # t2=time.time()
    # print('\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0))
    # print('norm = %g'%(norm(b - A*y)))
    # print('info flag = %d'%(flag))
