"""
Implémentation en Python de l'algorithme de stabilisation du gradient bi-conjugué (Bi-CGSTAB).
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.bicgstab.html

"""
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
import time
import sys
import petsc4py
petsc4py.init(sys.argv)
from numpy import array, inner, conjugate, ravel, diag, dot
from pyamg.util.linalg import norm, cond
from scipy.sparse.linalg.isolve.utils import make_system
from mpi4py import MPI
from petsc4py import PETSc

class BiCGSTAB:
    def __init__(self,**kwargs):
        pass

    def solve(A, b, tol, x=None, it_max=None, prec_A=None, callback=None, residuals=None):
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
            x : {array, matrix}
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
        Output :
            x_final : solution approchée
            info : statut d'arrêt du bicgstab
            r : résidu absolu
        -----------------------
        References
        .. [1] Yousef Saad, "Méthodes itératives pour les systèmes linéaires "Sparse",
        Second Edition", SIAM, pp. 231-234, 2003
        http://www-users.cs.umn.edu/~saad/books.html

        =======================
        """
        # Convertir les entrées en système linéaire
        (A, prec_A, x, b, postprocess) = make_system(A, prec_A, x, b)

        # Vérifie le nombre d'itérations
        if it_max == None:
            it_max = len(x) + 5
        elif it_max < 1:
            raise ValueError('Le nombre d itérations doit être positif')

        # Préparation
        r = b - A*x
        # Calcule de la norme de r
        normr = norm(r)

        # Défini les résidus initiaux si les résidus sont vides
        if residuals is None:
            residuals = []
        residuals.append(normr)

        # Calcule et Vérifie la norme b
        normb = norm(b)
        if normb == 0.0:
            normb = 1.0

        # Condition d'arrêt
        if normr < tol*normb:
            return ((x),0,norm(b - A*x)/norm(b))

        # Copies de r
        r_copy = r.copy()
        p = r.copy()

        # Produit interne ordinaire de vecteurs pour les tableaux 1-D (sans conjugaison complexe)
        # Le conjugué complexe d'un nombre complexe est obtenu en changeant le signe de sa partie imaginaire
        # Le produit d'un complexe et de son conjugué est égal au carré du module.
        delta = inner(r_copy.conjugate(), r)

        # Configure l'itération à zéro
        iter = 0

        # Boucle
        while True:

            # Condition du préconditionneur
            if prec_A is not None:
                A_matrix_p = A*prec_A*p
                matrix_p = prec_A*p
                pass
            else :
                A_matrix_p = A*p
                matrix_p = p
                pass

            # alpha = (r_{iter}, r_copy) / (A*p_{iter}, r_copy)
            alpha = delta/inner(r_copy.conjugate(), A_matrix_p)

            # s_{iter} = r_{iter} - alpha*A*p_{iter}
            s   = r - alpha*A_matrix_p

            # Condition du préconditionneur
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

            # beta_{iter} = (r_{iter+1}, r_copy)/(r_{iter}, r_copy) * (alpha/omega)
            delta_new = inner(r_copy.conjugate(), r)
            beta = (delta_new / delta) * (alpha / omega)
            delta = delta_new

            # p_{iter+1} = r_{iter+1} + beta*(p_{iter} - omega*A*p)
            p = r + beta*(p - omega*A_matrix_p)

            # +1
            iter += 1

            # Calcule la nouvelle norme de r
            normr = norm(r)

            # Ajout de la nouvelle norme de r aux résidus
            residuals.append(normr)

            # Appels des callback si ils existent
            if callback is not None:
                callback(x)

            # Condition d'arrêt
            if normr < tol:
                print('iter :',iter)
                return ((x), 0,norm(b - A*x)/norm(b))

            # Condition d'arrêt
            if iter == it_max:
                return ((x), iter,norm(b - A*x)/norm(b))
            pass
        pass


# Main - test
if __name__ == '__main__':
    # Initialisation de la taille de notre matrice
    n = 100

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
    ksp.setTolerances(1e-12)
    ksp.setIterationNumber(10000)
    ksp.setOperators(A_linalg)

    # conditionnement de la Matrice de systèmes linéaires à résoudre
    print('conditionnement =',cond(A))

    # Début du test
    print('\nTesting BiCGStab')

    # Self - Test
    t1=time.time()
    (x,flag,r) = BiCGSTAB.solve(A=A,b=b,tol=1e-12,x=x,it_max=10000)
    t2=time.time()
    print('\n%s took %0.3f ms' % ('bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%r )
    print('info flag = %d'%(flag))

    print("\n")
    # PETSc - Test
    t1=time.time()
    ksp.solve(b_linalg, x_linalg)
    t2=time.time()
    print('\n%s took %0.3f ms' % ('linalg bicgstab', (t2-t1)*1000.0))
    print('residu absolu = %g'%(norm(b_linalg - A_linalg*x_linalg)/norm(b_linalg)) )
