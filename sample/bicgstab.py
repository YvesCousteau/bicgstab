"""
Implémentation en Python de l'algorithme de stabilisation du gradient bi-conjugué (Bi-CGSTAB).
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.bicgstab.html

"""
import numpy as np
from numpy import inner, conjugate
from pyamg.util.linalg import norm
from scipy.sparse.linalg.isolve.utils import make_system
import matplotlib
from matplotlib import pyplot as plt
import time

def bicgstab_solver(A, b, tol, x=None, it_max=None, prec_A=None, callback=None, residuals=None):
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
    r = (b - A*x)
    # Calcule de la norme de r
    normr = (norm(r)/norm(b))
    # Défini les résidus initiaux si les résidus sont vides
    if residuals is None:
        residuals = []

    # Calcule et Vérifie la norme b
    normb = (norm(b))
    if normb == 0.0:
        normb = 1.0
    # Condition d'arrêt
    if normr < tol*normb:
        return ((x),0,(norm(b - A*x)/norm(b)),0,residuals)
    # Copies de r
    r_copy = r.copy()
    p = r.copy()
    # Produit interne ordinaire de vecteurs pour les tableaux 1-D (sans conjugaison complexe)
    # Le conjugué complexe d'un nombre complexe est obtenu en changeant le signe de sa partie imaginaire
    # Le produit d'un complexe et de son conjugué est égal au carré du module.
    delta = (inner(r_copy.conjugate(), r))
    # Configure l'itération à zéro
    iter = 0
    error = 0
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
        alpha = (delta/inner(r_copy.conjugate(), A_matrix_p))

        # s_{iter} = r_{iter} - alpha*A*p_{iter}
        s   = (r - alpha*A_matrix_p)
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
        omega = (inner(A_matrix_s.conjugate(), s)/inner(A_matrix_s.conjugate(), A_matrix_s))
        # x_{iter+1} = x_{iter} +  alpha*p_{iter} + omega*s_{iter}
        x = (x + alpha*matrix_p + omega*matrix_s)
        # r_{iter+1} = s_{iter} - omega*A*s
        r = (s - omega*A_matrix_s)
        # beta_{iter} = (r_{iter+1}, r_copy)/(r_{iter}, r_copy) * (alpha/omega)
        delta_new = (inner(r_copy.conjugate(), r))
        if (delta == 0):
            print('delta')
            return ((x), 0,norm(b - A*x)/norm(b),iter,residuals)
        if (omega == 0):
            print('omega')
            return ((x), 0,norm(b - A*x)/norm(b),iter,residuals)
        beta = ((delta_new / delta) * (alpha / omega))
        delta = (delta_new)
        # p_{iter+1} = r_{iter+1} + beta*(p_{iter} - omega*A*p)
        p = (r + beta*(p - omega*A_matrix_p))
        # +1
        iter += 1
        # Calcule la nouvelle norme de r

        # Condition d'arrêt
        if int((norm(r) - normr)/tol) == 0:
            error += 1
            if error == 10:
                return ((x), 0,(norm(b - A*x)/norm(b)),iter,residuals)
        else:
            error = 0

        normr = (norm(r))
        # Ajout de la nouvelle norme de r aux résidus
        residuals.append(normr)
        # Appels des callback si ils existent
        if callback is not None:
            callback(x)
        # Condition d'arrêt
        if normr < tol:
            return ((x), 0,(norm(b - A*x)/norm(b)),iter,residuals)
        # Condition d'arrêt
        if iter == it_max:
            return ((x), iter,(norm(b - A*x)/norm(b)),iter,residuals)
        pass
    pass

def bcgs(mat_A,mat_b,mat_x,n,tol,it_max,cond,yaml_data):
    # Self - Initialisations
    # Création de la solution du vecteur x.
    x = []
    # Initialisations des vecteurs b et x
    for i in range(n):
        x.append(0)
    # Début du test
    print('\nTesting BiCGStab')
    # Self - Test
    t1=time.time()
    (x,flag,r,iter,conv_history) = bicgstab_solver(A=mat_A,b=mat_b,tol=tol,x=x,it_max=it_max)
    t2=time.time()
    print((t2-t1)*1000.0)
    residual = str((norm(mat_b - mat_A*x)/norm(mat_b)))
    forward_error = str((norm(mat_x - x)/norm(mat_x)))
    cond_backward_error = str((cond*((norm(mat_b - mat_A*x)/norm(mat_b)))))
    yaml_data["algo"] = {
        "solving time (s)":(t2-t1)*1000.0,
        'iter':iter,
        "vector x (norm)":str(norm(x)),
        'residual relative':residual,
        "forward error":forward_error,
        "condition * backward error":cond_backward_error,
    }
    plt.semilogy(conv_history, label="algo")
    # plt.semilogy(x, label="x")
    return(x)
