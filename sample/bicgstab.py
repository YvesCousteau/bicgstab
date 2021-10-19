"""
Implémentation en Python de l'algorithme de stabilisation du gradient bi-conjugué (Bi-CGSTAB).
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.bicgstab.html

"""

from numpy import array, inner, conjugate, ravel
from pyamg.util.linalg import norm

class BiCGSTAB:
    def __init__(self,**kwargs):
        pass

    def solve(self,A, b, x_init=None, it_max=None, tol=1e-5, prec_A=None, callback=None, residuals=None, xtype=None):
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
                Supposition de départ pour la solution (initialement a None)
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
        A,prec_A,x_init,b,postprocess = make_system(A,prec_A,x_init,b,xtype)

        # Check iteration numbers
        if it_max == None:
            it_max = len(x_init) + 5
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
            return (postprocess(x_init),0)

        # Raise  tol if norm r isn't small
        if normr != 0.0:
            tol = tol*normr

        # Check if matrix A is on 1 dimension
        # ravel => A 1-D array, containing the elements of the input, is returned.
        if A.shape[0] == 1:
            entry = ravel(A*array([1.0], dtype=xtype))
            return (postprocess(b/entry), 0)

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

            # alpha = (r_iter, rstar) / (A*p_iter, rstar)
            alpha = delta/inner(rstar.conjugate(), A_matrix_p)

            # s_iter = r_iter - alpha*A*p_iter
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

            # omega = (A*s_iter, s_iter)/(A*s_iter, A*s_iter)
            omega = inner(A_matrix_s.conjugate(), s)/inner(A_matrix_s.conjugate(), A_matrix_s)

            # x_{iter+1} = x_iter +  alpha*p_iter + omega*s_iter
            x = x + alpha*matrix_p + omega*matrix_s

            # r_{iter+1} = s_iter - omega*A*s
            r = s - omega*A_matrix_s

            # beta_iter = (r_{iter+1}, rstar)/(r_iter, rstar) * (alpha/omega)
            delta_new = inner(rstar.conjugate(), r)
            beta = (delta_new / delta) * (alpha / omega)
            delta = delta_new

            # p_{iter+1} = r_{iter+1} + beta*(p_iter - omega*A*p)
            p = r + beta*(p - omega*A_matrix_p)

            iter += 1

            normr = norm(r)

            # Stopping conditions
            if residuals is not None:
                residuals.append(normr)

            if callback is not None:
                callback(x)

            if normr < tol:
                return (postprocess(x), 0)

            if iter == maxiter:
                return (postprocess(x), iter)

            pass
        pass
