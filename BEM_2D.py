#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt


n = None
L = None
h = None
a = None
eps = None
x = None

E1 = None
nu1 = None
G1 = None

E2 = None
nu2 = None
G2 = None


f   = lambda x, y, nu: -1/(4*np.pi*(1-nu))*(y*(np.arctan2(y,x-a)-np.arctan2(y,x+a))-(x-a)*np.log(np.sqrt((x-a)**2+y**2))+(x+a)*np.log(np.sqrt((x+a)**2+y**2)))
fx  = lambda x, y, nu:  1/(4*np.pi*(1-nu))*(np.log(np.sqrt((x-a)**2+y**2))-np.log(np.sqrt((x+a)**2+y**2)))
fy  = lambda x, y, nu: -1/(4*np.pi*(1-nu))*(np.arctan2(y,x-a)-np.arctan2(y,x+a))
fxy = lambda x, y, nu:  1/(4*np.pi*(1-nu))*(y/((x-a)**2+y**2)-y/((x+a)**2+y**2))
fxx = lambda x, y, nu:  1/(4*np.pi*(1-nu))*((x-a)/((x-a)**2+y**2)-(x+a)/((x+a)**2+y**2))
fyy = lambda x, y, nu: -fxx(x, y, nu)

uxx = lambda x, y, G, nu:  1/(2*G)*((3-4*nu)*f(x,y,nu)+y*fy(x,y,nu))
uxy = lambda x, y, G, nu:  1/(2*G)*y*fx(x,y,nu) # Changed sign from book??
uyx = lambda x, y, G, nu: -1/(2*G)*y*fx(x,y,nu)
uyy = lambda x, y, G, nu:  1/(2*G)*((3-4*nu)*f(x,y,nu)-y*fy(x,y,nu))

sxxx = lambda x, y, nu:  (3-2*nu)*fx(x,y,nu)+y*fxy(x,y,nu)
sxxy = lambda x, y, nu:      2*nu*fy(x,y,nu)+y*fyy(x,y,nu)
sxyx = lambda x, y, nu:  2*(1-nu)*fy(x,y,nu)+y*fyy(x,y,nu)
sxyy = lambda x, y, nu:  (1-2*nu)*fx(x,y,nu)-y*fxy(x,y,nu)
syyx = lambda x, y, nu: -(1-2*nu)*fx(x,y,nu)-y*fxy(x,y,nu)
syyy = lambda x, y, nu:  2*(1-nu)*fy(x,y,nu)-y*fyy(x,y,nu)


ya = None
pa = None
Kinv = None
K = None
Uc = None
Pc = None
Qa = None
Qb = None
Qc = None

interface = "frictionless" # "frictionless" or "sticking"


def setGeometry(_n, _L, _h):
	global n, L, h, a, eps, x, ya
	n = _n
	L = _L
	h = _h
	a = L / (n - 1) / 2
	eps = a * 1e-10
	x = np.linspace(-L/2 + a, L/2 - a, n)[:, np.newaxis]
	if ya is None or ya.shape[0] != n:
		ya = np.zeros((n, 1))
	updateStiffnessMatrix()


def setSurfaceTop(y1):
	global ya
	ya = y1
	updateStiffnessMatrix()


def setMaterialTop(E1_, nu1_):
	global E1, nu1, G1
	E1 = E1_
	nu1 = nu1_
	G1 = E1 / (2 * (1 + nu1))
	updateStiffnessMatrix()


def setMaterialBottom(E2_, nu2_):
	global E2, nu2, G2
	E2 = E2_
	nu2 = nu2_
	G2 = E2 / (2 * (2 + nu2))
	updateStiffnessMatrix()


def updateStiffnessMatrix():
	global Kinv, K, Uc, Pc, Qa, Qb, Qc
	if not initialized: return
	
	x1 = np.block([[x], [x]])
	yb = -h * np.ones((n, 1))
	y1 = np.block([[ya], [yb]])
	n1 = np.block([[np.ones((n, 1))], [-np.ones((n, 1))]])
	x2 = x
	y2 = yb
	n2 = np.ones((n, 1))
	
	U1xx = np.zeros((2*n, 2*n))
	U1xy = np.zeros((2*n, 2*n))
	U1yx = np.zeros((2*n, 2*n))
	U1yy = np.zeros((2*n, 2*n))
	U2xx = np.zeros((n, n))
	U2xy = np.zeros((n, n))
	U2yx = np.zeros((n, n))
	U2yy = np.zeros((n, n))
	P1xx = np.zeros((2*n, 2*n))
	P1xy = np.zeros((2*n, 2*n))
	P1yx = np.zeros((2*n, 2*n))
	P1yy = np.zeros((2*n, 2*n))
	P2xx = np.zeros((n, n))
	P2xy = np.zeros((n, n))
	P2yx = np.zeros((n, n))
	P2yy = np.zeros((n, n))

	for i in range(2*n): # loop on q_i
		U1xx[:, i] = uxx(x1 - x1[i],  y1 - y1[i], G1, nu1).flatten()
		U1xy[:, i] = uxy(x1 - x1[i],  y1 - y1[i], G1, nu1).flatten()
		U1yx[:, i] = uyx(x1 - x1[i],  y1 - y1[i], G1, nu1).flatten()
		U1yy[:, i] = uyy(x1 - x1[i],  y1 - y1[i], G1, nu1).flatten()
		P1xx[:, i] = (sxyx(x1 - x1[i],  y1 - y1[i] - eps * n1, nu1) * n1).flatten()
		P1xy[:, i] = (sxyy(x1 - x1[i],  y1 - y1[i] - eps * n1, nu1) * n1).flatten()
		P1yx[:, i] = (syyx(x1 - x1[i],  y1 - y1[i] - eps * n1, nu1) * n1).flatten()
		P1yy[:, i] = (syyy(x1 - x1[i],  y1 - y1[i] - eps * n1, nu1) * n1).flatten()

	for i in range(n): # loop on q_i
		U2xx[:, i] = uxx(x2 - x2[i],  y2 - y2[i], G2, nu2).flatten()
		U2xy[:, i] = uxy(x2 - x2[i],  y2 - y2[i], G2, nu2).flatten()
		U2yx[:, i] = uyx(x2 - x2[i],  y2 - y2[i], G2, nu2).flatten()
		U2yy[:, i] = uyy(x2 - x2[i],  y2 - y2[i], G2, nu2).flatten()
		P2xx[:, i] = (sxyx(x2 - x2[i],  y2 - y2[i] - eps * n2, nu2) * n2).flatten()
		P2xy[:, i] = (sxyy(x2 - x2[i],  y2 - y2[i] - eps * n2, nu2) * n2).flatten()
		P2yx[:, i] = (syyx(x2 - x2[i],  y2 - y2[i] - eps * n2, nu2) * n2).flatten()
		P2yy[:, i] = (syyy(x2 - x2[i],  y2 - y2[i] - eps * n2, nu2) * n2).flatten()

	Uaa = np.block([[U1xx[  0:  n,   0:  n], U1xy[  0:  n,   0:  n]], [U1yx[  0:  n,   0:  n], U1yy[  0:  n,   0:  n]]])
	Uab = np.block([[U1xx[  0:  n,   n:2*n], U1xy[  0:  n,   n:2*n]], [U1yx[  0:  n,   n:2*n], U1yy[  0:  n,   n:2*n]]])
	Uba = np.block([[U1xx[  n:2*n,   0:  n], U1xy[  n:2*n,   0:  n]], [U1yx[  n:2*n,   0:  n], U1yy[  n:2*n,   0:  n]]])
	Ubb = np.block([[U1xx[  n:2*n,   n:2*n], U1xy[  n:2*n,   n:2*n]], [U1yx[  n:2*n,   n:2*n], U1yy[  n:2*n,   n:2*n]]])
	Ucc = np.block([[U2xx, U2xy], [U2yx, U2yy]])
	Paa = np.block([[P1xx[  0:  n,   0:  n], P1xy[  0:  n,   0:  n]], [P1yx[  0:  n,   0:  n], P1yy[  0:  n,   0:  n]]])
	Pab = np.block([[P1xx[  0:  n,   n:2*n], P1xy[  0:  n,   n:2*n]], [P1yx[  0:  n,   n:2*n], P1yy[  0:  n,   n:2*n]]])
	Pba = np.block([[P1xx[  n:2*n,   0:  n], P1xy[  n:2*n,   0:  n]], [P1yx[  n:2*n,   0:  n], P1yy[  n:2*n,   0:  n]]])
	Pbb = np.block([[P1xx[  n:2*n,   n:2*n], P1xy[  n:2*n,   n:2*n]], [P1yx[  n:2*n,   n:2*n], P1yy[  n:2*n,   n:2*n]]])
	Pcc = np.block([[P2xx, P2xy], [P2yx, P2yy]])

	M = None

	if interface == "sticking":
		# # Compact form:
		# M = np.block([
		# 	[Paa, Pab, np.zeros((2*n, 2*n))], # imposed surface pressure
		# 	[Pba, Pbb, Pcc],  # same stresses at interface
		# 	[Uba, Ubb, -Ucc], # same displacements at interface
		# ])

		# Separated x and y components
		M = np.block([
			[Paa, Pab, np.zeros((2*n, 2*n))],      # imposed surface pressure
			[Pba[:n, :], Pbb[:n, :], Pcc[:n, :]],  # same stresses in x direction at interface
			[Pba[n:, :], Pbb[n:, :], Pcc[n:, :]],  # same stresses in y direction at interface
			[Uba[:n, :], Ubb[:n, :], -Ucc[:n, :]], # same displacements in x direction at interface
			[Uba[n:, :], Ubb[n:, :], -Ucc[n:, :]], # same displacements in y direction at interface
		])
	
	elif interface == "frictionless":
		# Frictionless
		M = np.block([
			[Paa, Pab, np.zeros((2*n, 2*n))],                     # imposed surface pressure
			[Pba[:n, :], Pbb[:n, :], np.zeros((n, 2*n))],         # 0 tangential stresses under top layer
			[np.zeros((n, 2*n)), np.zeros((n, 2*n)), Pcc[:n, :]], # 0 tangential stresses over bottom layer
			[Pba[n:, :], Pbb[n:, :], Pcc[n:, :]],                 # same stresses in y direction at interface
			[Uba[n:, :], Ubb[n:, :], -Ucc[n:, :]],                # same displacements in y direction at interface
		])
	
	else:
		print("Unknown interface type " + interface)
		return
	
	Minv = np.linalg.inv(M)
	Qa = Minv[:2*n,    :2*n]
	Qb = Minv[2*n:4*n, :2*n]
	Qc = Minv[4*n:,  :2*n]
	
	Kinv = Uaa @ Qa + Uab @ Qb
	K = np.linalg.inv(Kinv)
	Uc = Ucc @ Qc
	Pc = Pcc @ Qc


def solve(pax, pay):
	global pa
	pa = np.block([[pax], [pay]])

	ua = Kinv @ pa
	uc = Uc @ pa
	pc = Pc @ pa

	qa = Qa @ pa
	# plt.figure()
	# plt.plot(x, qa[:n], "C0--")
	# plt.plot(x, qa[n:], "C0")
	# plt.show(block=False)

	uax = ua[:n]
	uay = ua[n:]
	ucx = uc[:n]
	ucy = uc[n:]
	pcx = pc[:n]
	pcy = pc[n:]
	return x, uax, uay, ucx, ucy, pcx, pcy


def getVolumeDisplacements(xg, yg):
	xx, yy = np.meshgrid(xg, yg)
	xx_flat = xx.flatten()[:, None]
	yy_flat = yy.flatten()[:, None]
	ny = len(yg)
	nt = xx_flat.shape[0]

	yb = -h * np.ones((n, 1))
	yc = yb

	Uaxx = np.zeros((nt, n))
	Uaxy = np.zeros((nt, n))
	Uayx = np.zeros((nt, n))
	Uayy = np.zeros((nt, n))
	Ubxx = np.zeros((nt, n))
	Ubxy = np.zeros((nt, n))
	Ubyx = np.zeros((nt, n))
	Ubyy = np.zeros((nt, n))
	Ucxx = np.zeros((nt, n))
	Ucxy = np.zeros((nt, n))
	Ucyx = np.zeros((nt, n))
	Ucyy = np.zeros((nt, n))

	for i in range(n): # loop on q_i
		Uaxx[:, i] = uxx(xx_flat - x[i], yy_flat - ya[i], G1, nu1).flatten()
		Uaxy[:, i] = uxy(xx_flat - x[i], yy_flat - ya[i], G1, nu1).flatten()
		Uayx[:, i] = uyx(xx_flat - x[i], yy_flat - ya[i], G1, nu1).flatten()
		Uayy[:, i] = uyy(xx_flat - x[i], yy_flat - ya[i], G1, nu1).flatten()
		Ubxx[:, i] = uxx(xx_flat - x[i], yy_flat - yb[i], G1, nu1).flatten()
		Ubxy[:, i] = uxy(xx_flat - x[i], yy_flat - yb[i], G1, nu1).flatten()
		Ubyx[:, i] = uyx(xx_flat - x[i], yy_flat - yb[i], G1, nu1).flatten()
		Ubyy[:, i] = uyy(xx_flat - x[i], yy_flat - yb[i], G1, nu1).flatten()
		Ucxx[:, i] = uxx(xx_flat - x[i], yy_flat - yc[i], G2, nu2).flatten()
		Ucxy[:, i] = uxy(xx_flat - x[i], yy_flat - yc[i], G2, nu2).flatten()
		Ucyx[:, i] = uyx(xx_flat - x[i], yy_flat - yc[i], G2, nu2).flatten()
		Ucyy[:, i] = uyy(xx_flat - x[i], yy_flat - yc[i], G2, nu2).flatten()

	Ua = np.block([[Uaxx, Uaxy], [Uayx, Uayy]])
	Ub = np.block([[Ubxx, Ubxy], [Ubyx, Ubyy]])
	Uc = np.block([[Ucxx, Ucxy], [Ucyx, Ucyy]])

	qa = Qa @ pa
	qb = Qb @ pa
	qc = Qc @ pa

	top = yy_flat > np.repeat(yb, ny, axis=0)
	top = np.block([[top], [top]])

	u = top * (Ua @ qa + Ub @ qb) + np.invert(top) * (Uc @ qc) # TODO: update to take yb instead of -h
	u = np.reshape(u, (2*ny, n))
	ux = u[:ny, :]
	uy = u[ny:, :]
	
	return xx, yy, ux, uy


def getVolumeStresses(xg, yg):
	xx, yy = np.meshgrid(xg, yg)
	xx_flat = xx.flatten()[:, None]
	yy_flat = yy.flatten()[:, None]
	ny = len(yg)
	nt = xx_flat.shape[0]

	yb = -h * np.ones((n, 1))
	yc = yb

	Saxxx = np.zeros((nt, n))
	Saxxy = np.zeros((nt, n))
	Sbxxx = np.zeros((nt, n))
	Sbxxy = np.zeros((nt, n))
	Scxxx = np.zeros((nt, n))
	Scxxy = np.zeros((nt, n))
	Saxyx = np.zeros((nt, n))
	Saxyy = np.zeros((nt, n))
	Sbxyx = np.zeros((nt, n))
	Sbxyy = np.zeros((nt, n))
	Scxyx = np.zeros((nt, n))
	Scxyy = np.zeros((nt, n))
	Sayyx = np.zeros((nt, n))
	Sayyy = np.zeros((nt, n))
	Sbyyx = np.zeros((nt, n))
	Sbyyy = np.zeros((nt, n))
	Scyyx = np.zeros((nt, n))
	Scyyy = np.zeros((nt, n))


	for i in range(n): # loop on q_i
		Saxxx[:, i] = sxxx(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Saxxy[:, i] = sxxy(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Sbxxx[:, i] = sxxx(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Sbxxy[:, i] = sxxy(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Scxxx[:, i] = sxxx(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()
		Scxxy[:, i] = sxxy(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()
		Saxyx[:, i] = sxyx(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Saxyy[:, i] = sxyy(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Sbxyx[:, i] = sxyx(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Sbxyy[:, i] = sxyy(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Scxyx[:, i] = sxyx(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()
		Scxyy[:, i] = sxyy(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()
		Sayyx[:, i] = syyx(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Sayyy[:, i] = syyy(xx_flat - x[i], yy_flat - ya[i] - eps, nu1).flatten()
		Sbyyx[:, i] = syyx(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Sbyyy[:, i] = syyy(xx_flat - x[i], yy_flat - yb[i] + eps, nu1).flatten()
		Scyyx[:, i] = syyx(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()
		Scyyy[:, i] = syyy(xx_flat - x[i], yy_flat - yc[i] - eps, nu2).flatten()

	Saxx = np.block([[Saxxx, Saxxy]])
	Sbxx = np.block([[Sbxxx, Sbxxy]])
	Scxx = np.block([[Scxxx, Scxxy]])
	Saxy = np.block([[Saxyx, Saxyy]])
	Sbxy = np.block([[Sbxyx, Sbxyy]])
	Scxy = np.block([[Scxyx, Scxyy]])
	Sayy = np.block([[Sayyx, Sayyy]])
	Sbyy = np.block([[Sbyyx, Sbyyy]])
	Scyy = np.block([[Scyyx, Scyyy]])

	qa = Qa @ pa
	qb = Qb @ pa
	qc = Qc @ pa

	top = yy_flat > np.repeat(yb, ny, axis=0)
	sxx = top * (Saxx @ qa + Sbxx @ qb) + np.invert(top) * (Scxx @ qc) # TODO: update to take yb instead of -h
	sxy = top * (Saxy @ qa + Sbxy @ qb) + np.invert(top) * (Scxy @ qc)
	syy = top * (Sayy @ qa + Sbyy @ qb) + np.invert(top) * (Scyy @ qc)
	sxx = np.reshape(sxx, xx.shape)
	sxy = np.reshape(sxy, xx.shape)
	syy = np.reshape(syy, xx.shape)
	
	return xx, yy, sxx, syy, sxy


initialized = False
setGeometry(51, 1, 0.1)
setMaterialTop(1, 0.3)
setMaterialBottom(1, 0.3)
initialized = True
updateStiffnessMatrix()

