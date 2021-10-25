#!/usr/bin/env python


import numpy as np
from matplotlib import pyplot as plt
import BEM_2D as BEM
import math
from numpy.linalg import eig

Etest = [1/10, 1, 10]
Htest = [10, 0.5, 0.2]
## Geometric
n = 101
L = 1
x = np.linspace(-L / 2 * n / (n - 1), L / 2 * n / (n - 1), n)
## Top surface shape
y1 = np.zeros((n, 1))

nu1 = 0.3
E2 = 1
nu2 = 0.3

EShape = 100
nuShape = 0.3
RShape = 8
HShape = np.zeros((n, 1))
ainit = L / 10

for i in range(n):
    if abs(x[i]) <= RShape:
        HShape[i] = - math.sqrt(RShape ** 2 - x[i] ** 2)
    else:
        HShape[i] = 0
plt.figure()

for i1 in range(len(Etest)):
    for i2 in range(len(Htest)):
        h = Htest[i2]
        yH = np.zeros((n, 1))-h
        ## Materials
        E1 = Etest[i1]
        ## Initial Pressure
        p1x = np.zeros((n, 1))
        p1y = np.zeros((n, 1))
        P0x = 0
        P0y = 1E-3
        Ncontact = len(np.where(HShape < 0)[0])

        for i in range(n):
            p1y[i] = P0y / n

        ## Initialisation
        eps0 = 1E-13
        eps = 1
        delta = 0
        Gold = 1
        gij = np.zeros((n, 1))
        uij = np.zeros((n, 1))
        r = np.zeros((n, 1))
        Iol = np.zeros((n, 1))

        BEM.interface = "sticking"
        BEM.setGeometry(n, L, h)
        BEM.setSurfaceTop(y1)
        BEM.setMaterialBottom(E2, nu2)
        BEM.setMaterialTop(E1, nu1)
        K = BEM.Kinv
        Kreshape = K[n:, n:]

        ##Algorithme
        while eps >= eps0:
            uij = - Kreshape @ p1y
            gij = - uij + HShape
            Index = np.where(p1y > 0)[0]  ##trouve les noeuds en contact
            Nc = len(Index)  ##nombre de noeuds en contact
            g_ = 1 / Nc * sum(gij[Index])  ##différent sur les deux papiers Ic ou Ig
            gij = gij - g_
            G = sum(gij[Index] * gij[Index])
            tij = np.zeros((n, 1))
            tij[Index] = gij[Index] + (delta * (G / Gold)) * tij[Index]
            Gold = G
            r = Kreshape @ tij
            r_ = 1 / Nc * sum(r[Index])
            r = r - r_
            Tau = sum(gij[Index] * tij[Index]) / sum(r[Index] * tij[Index])
            p1y[Index] = p1y[Index] - Tau * tij[Index]

            for i in range(n):
                if p1y[i] < 0:
                    p1y[i] = 0
                else:
                    p1y[i] = p1y[i]

            for i in range(n):
                if p1y[i] == 0 and gij[i] < 0:
                    Iol[i] = True
                else:
                    Iol[i] = False

            if np.sum(Iol) == 0:
                delta = 1
            else:
                delta = 0

            for i in range(n):
                if Iol[i] == 1:
                    p1y[i] = p1y[i] - Tau * gij[i]
                else:
                    p1y[i] = p1y[i]

            P = L / n * np.sum(p1y)
            p1y = (P0y / P) * p1y
            eps = np.sum(abs(p1y) * abs(gij))

        ##BEM solve with new pressure
        BEM.interface = "sticking"
        BEM.setGeometry(n, L, h)
        BEM.setSurfaceTop(y1)
        BEM.setMaterialBottom(E2, nu2)
        BEM.setMaterialTop(E1, nu1)
        x, u1x, u1y, u2x, u2y, p2x, p2y = BEM.solve(p1x, -p1y)

        m = 50
        xg = np.linspace(-L / 2 * n / (n - 1), L / 2 * n / (n - 1), n)
        yg = np.linspace(0, -1, m)
        xx, yy, ux, uy = BEM.getVolumeDisplacements(xg, yg)
        _, _, sxx, syy, sxy = BEM.getVolumeStresses(xg, yg)


        # calcul val/vect propre
        T = np.zeros((2, 2))
        v = np.zeros((m * n, 2))
        Tmax = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                T[0, 0] = sxx[i, j]
                T[0, 1] = sxy[i, j]
                T[1, 0] = sxy[i, j]
                T[1, 1] = syy[i, j]
                vij, Vij = eig(T)
                if vij[0] < vij[1]:
                    tmax = abs(vij[1]-vij[0]) / 2
                    Tmax[i, j] = tmax
                else:
                    tmax = abs(vij[0] - vij[1]) / 2
                    Tmax[i, j] = tmax
        Neg_Tmax = int(3*n * m / 100)
        Tmaxmax = np.unique(Tmax)[-Neg_Tmax]
        ##print(Tmaxmax)
        '''
        plt.subplot(3, 3, 3*i1 + i2 + 1)
        plt.pcolormesh(xx, yy, sxx, vmin=-7E-3, vmax=1E-4)
        plt.colorbar(label=r"$\sigma_{xx}$")
        if h != 10:
            plt.plot(x, yH, 'k', linestyle='dashed')
        CS = plt.contour(xx, yy, sxx, [-1E-3, -0.5E-3, -1E-4, -1E-5, 0], colors=('b', 'g', 'r', 'c', 'k'))
        plt.title("E1=" + str(E1) + ' h=' + str(h))
        plt.xlabel("Distance x")
        plt.ylabel("Distance y")
        
        plt.subplot(3, 3, 3*i1 + i2 + 1)
        plt.pcolormesh(xx, yy, sxy, vmin=-5E-3, vmax=5E-3)
        plt.colorbar(label=r"$\sigma_{xy}$")
        CS = plt.contour(xx, yy, sxy, [-1E-3, -0.5E-3, -1E-4, -1E-5, 0], colors=('b', 'g', 'r', 'c', 'k'))
        if h != 10:
            plt.plot(x, yH, 'k', linestyle='dashed')
        plt.title("E1=" + str(E1) + ' h=' + str(h))
        plt.xlabel("Distance x")
        plt.ylabel("Distance y")
        '''

        ##Tmaxmax = np.max(Tmax[:, n//3:2*n//3])

        plt.subplot(3, 3, 3*i1 + i2 + 1)
        C1 = plt.pcolormesh(xx, yy, Tmax, vmin=0, vmax=3E-3, shading='auto')
        plt.colorbar(label='Tmax')
        plt.contour(xx, yy, Tmax, 10, colors='k', linewidths=0.5)
        CB = plt.contour(xx, yy, Tmax, levels=[Tmaxmax], colors='r')
        if h != 10:
            plt.plot(x, yH, 'k', linestyle='dashed')
        plt.title("E1=" + str(E1) + ' h=' + str(h))
        plt.xlabel("Distance x")
        plt.ylabel("Distance y")

        plt.subplots_adjust(top=0.910, bottom=0.05, left=0.1, right=0.9, hspace=0.3, wspace=0.25)
plt.suptitle('Tangential maximum Stress - ' + BEM.interface + ' interface', fontsize=28)
plt.show()










