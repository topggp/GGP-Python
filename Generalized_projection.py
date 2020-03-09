# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:45:15 2019

@author: João Matos
"""

#from __future__ import division
import sys

### Imports for calculations
import numpy as np
from numpy.lib import scimath
import numpy.matlib
import math

### Import for solver
import scipy
from scipy.sparse import diags # or use numpy: from numpy import diag as diags
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

### Imports for plots
from matplotlib import colors
import matplotlib.pyplot as plt

def main(nelx=None, nely=None, volfrac=None, settings=None, BC=None, maxiteration=None):

    stopping_criteria='change'  #Stopping Criteria:  'change'  'kktnorm'

    if 'Top_Rib' == BC:
        starting_guess = 'Top_Rib'  # Starting guess: Crosses, Top_Rib, ...
    else:
        starting_guess = 'Crosses'

    p={}
    if settings=='GGP':
        p['method']='MMC'   #Method:   'MMC'  'MNA'  'GP'
        q=2                 #q=1
        p['zp']=1           #parameter for p-norm/mean regularization
        p['alp']=1          #parameter for MMC
        p['epsi']=0.866     #parameter for MMC
        p['bet']=1e-3       #parameter for MMC
        minh=1              # minimum height of a component
        p['aggregation']='KSl'   #parameter for the aggregation function to be used
        #'IE'-Induced Exponential  'KS'-KS function  'KSl'-Lowerbound KS function 'p-norm'  'p-mean'
        p['ka']=10          #parameter for the aggregation constant
        p['saturation']=True     #switch for saturation
        ncx=1           #number of components in the x direction
        ncy=1           #number of components in the y direction
        Ngp=2           #number of Gauss point per sampling window
        R=0.25          #radius of the sampling window (infty norm)
        initial_d=.5    #component initial mass
    elif settings=='MMC':
        p['method']='MMC'   #'MMC'  'MNA'  'GP'
        q=2                 #q=1
        p['zp']=1           #parameter for p-norm/mean regularization
        p['alp']=1          #parameter for MMC
        p['epsi']=0.866     #parameter for MMC
        p['bet']=1e-3       #parameter for MMC
        minh=1              # minimum height of a component
        p['aggregation']='KS'   #parameter for the aggregation function to be used
        #'IE'-Induced Exponential  'KS'-KS function  'KSl'-Lowerbound KS function 'p-norm'  'p-mean'
        p['ka']=10          #parameter for the aggregation constant
        p['saturation']=False    #switch for saturation
        ncx=1               #number of components in the x direction
        ncy=1               #number of components in the y direction
        Ngp=2               #number of Gauss point per sampling window
        R=math.sqrt(3)/2    #radius of the sampling window (infty norm)
        initial_d=1         #component initial mass
    elif settings=='MNA':
        p['method']='MNA'    #'MMC'  'MNA'  'GP'
        q=1             #q=1
        p['zp']=1           #parameter for p-norm/mean regularization
        minh=1              # minimum height of a component
        p['sigma']=2        #parameter for MNA
        p['penalty']=3      #parameter for MNA
        p['gammav']=1       #parameter for GP
        p['gammac']=3       #parameter for GP
        p['aggregation']='KSl'    #parameter for the aggregation function to be used
        #'IE'-Induced Exponential  'KS'-KS function  'KSl'-Lowerbound KS function 'p-norm'  'p-mean'
        p['ka']=10           #parameter for the aggregation constant
        p['saturation']=False     #switch for saturation
        ncx=1               #number of components in the x direction
        ncy=1               #number of components in the y direction
        Ngp=1               #number of Gauss point per sampling window
        R=math.sqrt(1)/2    #radius of the sampling window (infty norm)
        initial_d=0.5       #component initial mass
    elif settings=='GP':
        p['method']='GP'    #'MMC'  'MNA'  'GP'
        q=1                 #q=1
        p['zp']=1           #parameter for p-norm/mean regularization
        p['deltamin']=1e-6  #parameter for GP
        p['r']=1.5          #parameter for GP
        minh=1              # minimum height of a component
        p['gammav']=1       #parameter for GP
        p['gammac']=3       #parameter for GP
        p['aggregation']='KSl'  #parameter for the aggregation function to be used
        #'IE'-Induced Exponential  'KS'-KS function  'KSl'-Lowerbound KS function 'p-norm'  'p-mean'
        p['ka']=10          #parameter for the aggregation constant
        p['saturation']=False   #switch for saturation
        ncx=1               #number of components in the x direction
        ncy=1               #number of components in the y direction
        Ngp=2               #number of Gauss point per sampling window
        R=math.sqrt(1)/2    #radius of the sampling window (infty norm)
        initial_d=0.5       #component initial mass
    else:
        print('settings string should be a valid entry: ''GGP'',''MMC'',''MNA'',''GP''') 
        sys.exit()

    ### MATERIAL PROPERTIES
    p['E0'] = 1
    p['Emin'] = 1e-6
    
    ### PREPARE FINITE ELEMENT ANALYSIS
    ndof = 2*(nelx+1)*(nely+1)
    KE = lk()
    
    # FE: Build the index vectors for the for coo matrix format.
    edofMat=np.zeros((nelx*nely,8),dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1=(nely+1)*elx+ely
            n2=(nely+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()
    
    # Define the nodal coordinates
    x=0
    Yy=np.zeros((nelx+1)*(nely+1))
    Xx=np.zeros((nelx+1)*(nely+1))
    for i in range(1,nelx+2):
        for j in range(1,nely+2):
            Yy[x]= j
            Xx[x]= i
            x+=1
    Yy=nely+1-Yy
    Xx=Xx-1
    
    # Compute the centroid coordinates
    xc = np.zeros(nelx * nely)
    yc = np.zeros(nelx * nely)
    for xi in range(nelx):
        xc[xi*nely:(xi+1)*nely]=xi+0.5
    for yi in range(nely):
        yc[np.arange(yi,nelx*nely,nely)]=yi+0.5
    yc = nely - yc
    
    centroid_coordinate=np.column_stack((xc,yc))
    a=-R
    b=R
    [gpc,wc]=lgwt(Ngp,a,b)
    [gpcx,gpcy]=np.meshgrid(gpc,gpc)
    gpcx=gpcx
    gpcy=gpcy
    wc=wc[np.newaxis]
    gauss_weight=wc * wc.T
    gpcx = np.kron([np.ones(len(centroid_coordinate))],np.ravel(gpcx.T)[np.newaxis].T).flatten()
    gpcy = np.kron([np.ones(len(centroid_coordinate))],np.ravel(gpcy.T)[np.newaxis].T).flatten()
    gauss_weight = np.kron([np.ones(len(centroid_coordinate))],np.ravel(gauss_weight.T)[np.newaxis].T).flatten()
    cc=np.kron(np.ones(Ngp**2),centroid_coordinate.T).T
    gauss_point = cc + np.concatenate([[gpcx], [gpcy]]).T
    ugp, idgp = np.unique(gauss_point, axis=0, return_inverse=True)
    
    ### DEFINE LOADS AND SUPPORTS
    if 'MBB' == BC:
        excitation_node = 0         #Node where the force is applied
        excitation_direction = 1    # 0 for x and 1 for y
        amplitude = - 1             #Amplitude of the force
        f = np.zeros((ndof, 1))     #Force array Initialization
        f[2*(excitation_node)+excitation_direction, 0] = amplitude  #Force array
        fixednodes = np.concatenate(([[np.where(Xx == min(Xx))], [(nelx + 1) * (nely + 1) - 1]]), axis=None)    #Fixed nodes
        fixed_dir = np.concatenate(([[np.ones(nely + 1)], [2]]),axis=None)
        fixeddofs = np.dot(2, (fixednodes)) + fixed_dir - 1     #Fixed degrees of freedom
        emptyelts = []      #Obligatory empty elements
        fullelts = []       #Obligatory full elements
    elif 'Short_Cantiliever' == BC:
        excitation_node = np.where(np.logical_and((Xx == max(Xx)), (Yy == np.fix(np.dot(0.5, min(Yy)) + np.dot(0.5, max(Yy))))))[0][0]      #Nodes where the force is applied
        excitation_direction = 1    # 0 for x and 1 for y
        amplitude = - 1             #Amplitude of the force
        f = np.zeros((ndof, 1))     #Force array Initialization
        f[2*excitation_node+excitation_direction, 0] = amplitude  #Force array
        fixednodes = np.kron([1, 1],np.where(Xx == min (Xx))[0])    #Fixed nodes
        fixed_dir = np.concatenate([[np.ones(nely + 1)], [np.dot(2, np.ones(nely + 1))]]).flatten()
        fixeddofs = np.dot(2, fixednodes) + fixed_dir - 1
        emptyelts = []      #Obligatory empty elements
        fullelts = []       #Obligatory full elements
    elif 'L-Shape' == BC:
        excitation_node = np.where(np.logical_and((Xx == max(Xx)), (Yy == np.fix(np.dot(0.5, min(Yy)) + np.dot(0.5, max(Yy))))))[0][0]      #Nodes where the force is applied
        excitation_direction = 1    # 0 for x and 1 for y
        amplitude = - 1             #Amplitude of the force
        f = np.zeros((ndof, 1))     #Force array Initialization
        f[2*excitation_node+excitation_direction, 0] = amplitude  #Force array
        fixednodes = np.kron([1, 1],np.where(Yy == max (Yy))[0])    #Fixed nodes
        fixed_dir = np.concatenate([[np.ones(nelx + 1)], [np.dot(2, np.ones(nelx + 1))]]).flatten()
        fixeddofs = np.dot(2, fixednodes) + fixed_dir - 1     #Fixed degrees of freedom
        emptyelts = np.where(np.logical_and(xc >= (max(Xx) + min(Xx)) / 2,yc >= ((max(Yy) + min(Yy)) / 2)))[0]      #Obligatory empty elements
        fullelts = []       #Obligatory full element
    elif 'Top_Rib' == BC:
        d = 3 * nelx / 8
        R = 6 * nely
        Rint1 = nely / 4
        Rint2 = nely / 4
        Rint3 = nely / 6
        a = np.floor((np.sqrt((R) ** 2 - (d) ** 2)) - R + nely)  # edges
        b = np.floor((np.sqrt((R) ** 2 - (nelx - d) ** 2)) - R + nely)
        ff = int(np.floor(((nelx / 4) + Rint1 + (((nelx / 4) - Rint1 - Rint2) / 2) - (nelx / 80) + 1)))  # white square dimensions
        g = int(np.floor(((nelx / 4) + Rint1 + (((nelx / 4) - Rint1 - Rint2) / 2) + (nelx / 80))))

        '''
        l_ini = np.sqrt(volfrac*nelx*nely/(0.5*nX*nY))     # initial length of component
        [xElts, yElts] = np.meshgrid(np.linspace(1 / (nX + 1) * nelx, nX / (nX + 1) * nelx, nX),np.linspace(1 / (nY + 1) * nely, nY / (nY + 1) * nely, nY))




        xElts = np.reshape(xElts.T, (1, xElts.size))
        yElts = np.reshape(yElts.T, (1, yElts.size))

        xElts = xElts(~isnan(xElts))
        yElts = yElts(~isnan(yElts))

        n_c = xElts.shape[1]
        x = np.concatenate((xElts,yElts,np.zeros((1, n_c)),l_ini * np.ones((2, n_c))), axis = 0)
        n_var = x.size
        x = np.reshape(x, n_var, 1)
        '''

        emptyelts = []  # Obligatory empty elements
        fullelts = []  # Obligatory full elements
        passivewhite = np.zeros((nely,nelx))

        ely2 = 0
        while ely2 < nely:                 ###white external round
            elx2 = 0
            while elx2 < nelx:
                if np.sqrt((ely2 - R)**2 + (elx2 - d)** 2) > R:
                    emptyelts.append(nely*elx2+ely2)
                    passivewhite[ely2, elx2] = 1
                    #x[ely2, elx2] = 0.001
                elx2 = elx2 + 1
            ely2 = ely2 + 1

        ely3 = 0
        while ely3 < nely:  ###white external round
            elx3 = 0
            while elx3 < nelx:
                if np.sqrt((ely3-((3*nely)/5))**2+(elx3-(nelx/4))**2) <= Rint1:
                    emptyelts.append(nely*elx3+ely3)
                    passivewhite[ely3, elx3] = 1
                    # x[ely3, elx3] = 0.001
                elx3 = elx3 + 1
            ely3 = ely3 + 1

        ely4 = 0
        while ely4 < nely:  ###white external round
            elx4 = 0
            while elx4 < nelx:
                if np.sqrt((ely4 - ((3 * nely) / 5)) ** 2 + (elx4 - (nelx / 2)) ** 2) <= Rint2:
                    emptyelts.append(nely*elx4+ely4)
                    passivewhite[ely4, elx4] = 1
                    # x[ely4, elx4] = 0.001
                elx4 = elx4 + 1
            ely4 = ely4 + 1

        ely5 = 0
        while ely5 < nely:  ###white external round
            elx5 = 0
            while elx5 < nelx:
                if np.sqrt((ely5 - ((3 * nely) / 5)) ** 2 + (elx5 - (3*nelx / 4)) ** 2) <= Rint3:
                    emptyelts.append(nely*elx5+ely5)
                    passivewhite[ely5, elx5] = 1
                    # x[ely5, elx5] = 0.001
                elx5 = elx5 + 1
            ely5 = ely5 + 1

        ely6 = int(nely / 10)
        while ely6 < 9 * nely / 10:  ###white external round
            elx6 = ff - 1
            while elx6 < g:
                emptyelts.append(nely * elx6 + ely6)
                passivewhite[ely6, elx6] = 1
                # x[ely6, elx6] = 0.001
                elx6 = elx6 + 1
            ely6 = ely6 + 1

        ###LOADS AND SUPPORTS
        dinx1 = int(2 * (nely - a + 1) - 1)                        # above
        diny1 = int(2 * (nely - a + 1))
        dinx2 = int(2 * ((nely + 1) * (d) + 1) - 1)
        diny2 = int(2 * ((nely + 1) * (d) + 1))
        dinx3 = int(2 * ((nely + 1) * (2 * d) + nely - a + 1) - 1)
        diny3 = int(2 * ((nely + 1) * (2 * d) + nely - a + 1))
        dinx4 = int(2 * ((nely + 1) * (nelx) + nely - b + 1) - 1)
        diny4 = int(2 * ((nely + 1) * (nelx) + nely - b + 1))

        dout1 = int(2 * ((nely + 1) * ff + (nely / 10) + 1))
        dout3 = int(2 * ((nely + 1) * g + (nely / 10) + 1))

        dinx5 = int(2 * ((nely + 1)) - 1)                           # below
        diny5 = int(2 * ((nely + 1)))
        dinx6 = int(2 * ((nely + 1) * (d + 1)) - 1)
        diny6 = int(2 * ((nely + 1) * (d + 1)))
        dinx7 = int(2 * ((nely + 1) * (2 * d + 1)) - 1)
        diny7 = int(2 * ((nely + 1) * (2 * d + 1)))
        dinx8 = int(2 * ((nely + 1) * (nelx + 1)) - 1)
        diny8 = int(2 * ((nely + 1) * (nelx + 1)))

        dout2 = int(2 * ((nely + 1) * ff + (9 * nely) / 10))
        dout4 = int(2 * ((nely + 1) * g + (9 * nely) / 10))

        f = np.zeros((ndof, 1))  # Force array Initialization
        f[dinx1] = 1                            # above
        f[diny1] = 1
        f[dinx2] = 1
        f[diny2] = 1
        f[dinx3] = 1
        f[diny3] = 1
        f[dinx4] = 1
        f[diny4] = 1

        f[dout1] = 1
        f[dout3] = 1

        f[dinx5] = -1                           # below
        f[diny5] = -1
        f[dinx6] = -1
        f[diny6] = -1
        f[dinx7] = -1
        f[diny7] = -1
        f[dinx8] = -1
        f[diny8 - 1] = -1

        f[dout2] = -1
        f[dout4] = -1

        fixeddofs1 = np.arange((2 * (nely - a + 1 + 1) - 1), (2 * (nely + 1 - 1) - 1))  # Fixed degrees of freedom
        fixeddofs2 = np.arange((2 * (nely - a + 1 + 1)), (2 * (nely + 1 - 1)))
        fixeddofs = np.concatenate([[fixeddofs1], [fixeddofs2]])

        '''            
        passivewhite=np.zeros((nely,nelx))
        passiveblack=np.zeros((nely,nelx))

        for ely2 in np.arange(1,nely):            # white external round
            for elx2 in np.arange(1,nelx):
                if np.sqrt((ely2-R) ** 2+(elx2-d) ** 2) > R:
                    passivewhite[ely2,elx2] = 1
                    #x[ely2,elx2] = 0.001
        for ely3 in np.arange(1,nely):            # white external round
            for elx3 in np.arange(1,nelx):
                if np.sqrt((ely3-((3*nely)/5)) ** 2+(elx3-(nelx/4)) ** 2) <= Rint1:
                    passivewhite[ely3,elx3] = 1
                    #x[ely3,elx3] = 0.001
        for ely4 in np.arange(1,nely):            # white external round
            for elx4 in np.arange(1,nelx):
                if np.sqrt((ely4-((3*nely)/5)) ** 2+(elx4-(nelx/2)) ** 2) <= Rint2:
                    passivewhite[ely4,elx4] = 1
                    #x[ely4,elx4] = 0.001
        for ely5 in np.arange(1,nely):            # white external round
            for elx5 in np.arange(1,nelx):
                if np.sqrt((ely5-((3*nely)/5)) ** 2+(elx5-((3*nelx)/4)) ** 2) <= Rint3:
                    passivewhite[ely5,elx5] = 1
                    #x[ely5,elx5] = 0.001
        for ely6 in np.arange(int((nely/10)+1),int((9*nely)/10)):
            for elx6 in np.arange(ff,g):
                passivewhite[ely6,elx6] = 1
                #x[elx6,elx6] = 0.001


        fixeddofs1 = np.arange((2 * (nely-a+1+1)-1), (2 * (nely+1-1)-1))    #Fixed degrees of freedom
        fixeddofs2 = np.arange((2 * (nely-a+1+1)),(2 * (nely+1-1)))
        fixeddofs = np.concatenate([[fixeddofs1],[fixeddofs2]])
        emptyelts = np.where(passivewhite == 1)[0]      #Obligatory empty elements
        fullelts = np.where(passiveblack == 1)[0]      #Obligatory full elements
        '''
    else:
        print("BC string should be a valid entry: 'MBB','L-Shape','Short_Cantiliever','Top_Rib'")
        sys.exit()
    
    alldofs = np.array(range(0,2*(nely+1)*(nelx+1)))
    freedofs = np.setdiff1d(alldofs, fixeddofs)         #Free degrees of freedom

    ## INITIALIZE ITERATION
    # define the initial guess components
    if starting_guess == 'Crosses': #cross_starting_guess
        xp = np.linspace(min(Xx), max(Xx), ncx + 2)
        yp = np.linspace(min(Yy), max(Yy), ncy + 2)
        xx, yy = np.meshgrid(xp, yp)
        Xc = np.kron([1, 1],np.ravel(xx[np.newaxis].T))
        Yc = np.kron([1, 1],np.ravel(yy[np.newaxis].T))
        Lc = np.dot(2 * np.sqrt((nelx/ (ncx + 2)) ** 2 + (nely/ (ncy + 2)) ** 2), np.ones(Xc.shape[0]))
        Tc = np.ravel(math.atan2(nely / ncy, nelx / ncx) * np.concatenate([[np.ones(int (Xc.shape[0]/2))], [-np.ones(int(Xc.shape[0]/2))]]))
        hc = 2 * np.ones(Xc.shape[0])
    elif starting_guess == 'Top_Rib':
        ##Inital Design
        ncx = 16     # number of components along x axis
        ncy = 4      # number of components along y axis
        nc = ncx * ncy      # number of components
        initial_d = 0.5       # component initial mass
        xp = np.linspace(1/(ncx+1)*nelx, ncx/(ncx+1)*nelx, ncx)
        yp = np.linspace(1/(ncy+1)*nely, ncy/(ncy+1)*nely, ncy)
        xx, yy = np.meshgrid(xp, yp)
        Xc = np.ravel(xx[np.newaxis].T)
        Yc = np.ravel(yy[np.newaxis].T)
        Lc = np.ones(Xc.shape[0]) * np.sqrt(volfrac * nelx * nely / (0.5 * ncx * ncy))  # initial length of component
        Tc = np.zeros(Xc.shape[0]) * Xc
        hc = Lc
    else:
        xp = np.linspace(min(Xx), max(Xx), ncx + 2)
        yp = np.linspace(min(Yy), max(Yy), ncy + 2)
        xx, yy = np.meshgrid(xp, yp)
        Xc = np.ravel(xx[np.newaxis].T)
        Yc = np.ravel(yy[np.newaxis].T)
        initial_Lh_ratio = nelx / nely
        hc = np.kron(np.ones(Xc.shape[0]),np.sqrt(1/initial_Lh_ratio*(nelx*nely)/ncx/ncy))
        Lc = initial_Lh_ratio * hc
        Tc = 0 * np.pi / 4 * np.ones(ncx * ncy)
    Mc = initial_d * np.ones(Xc.shape[0])

    # initial guess vector
    Xg = np.ravel(np.concatenate(([[Xc], [Yc], [Lc], [hc], [Tc], [Mc]])).T)

    ### Lower and Upper Limits
    Xl = min(Xx - 1) * np.ones(Xc.shape[0])
    Xu = max(Xx + 1) * np.ones(Xc.shape[0])
    Yl = min(Yy - 1) * np.ones(Xc.shape[0])
    Yu = max(Yy + 1) * np.ones(Xc.shape[0])
    Ll = 0 * np.ones(Xc.shape[0])
    Lu = np.sqrt(nelx ** 2 + nely ** 2) * np.ones(Xc.shape[0])
    hl = minh * np.ones(Xc.shape[0])
    hu = np.sqrt(nelx ** 2 + nely ** 2) * np.ones(Xc.shape[0])
    Tl = - 2 * np.pi * np.ones(Xc.shape[0])
    Tu = 2 * np.pi * np.ones(Xc.shape[0])
    Ml = 0 * np.ones(Xc.shape[0])
    Mu = np.ones(Xc.shape[0])
    if BC == 'Top_Rib':
        lower_bound = np.tile(np.concatenate(([0], [0], [0], [0], [-np.pi], [0])), nc)
        upper_bound = np.tile(np.concatenate(([nelx], [nely],[nelx], [nely], [np.pi], [1])), nc)
        X = (Xg - lower_bound) / (upper_bound - lower_bound)
    else:
        lower_bound = np.ravel(np.concatenate(([[Xl], [Yl], [Ll], [hl], [Tl], [Ml]])).T)
        upper_bound = np.ravel(np.concatenate(([[Xu], [Yu], [Lu], [hu], [Tu], [Mu]])).T)
        X = (Xg - lower_bound) / (upper_bound - lower_bound)
    
    m = 1
    n = X.shape[0]
    xval = X
    xold1 = xval
    xold2 = xval
    xmin = np.zeros(n)
    xmax = np.ones(n)
    low = np.zeros(n)
    upp = np.ones(n)
    C = 1000 * np.ones(m)
    d = 0 * np.ones(m)
    a0 = 1
    a = np.zeros(m)
    maxiteration = 300
    kkttol = 0.001
    changetol = 0.001
    kktnorm = kkttol + 10
    iteration = 0
    change = 1

    ### START ITERATION
    cvec = np.zeros(maxiteration)       #Vector with objective function for every iteration
    vvec = np.zeros(maxiteration)       #Vector with the volume fraction for every iteration
    kktnormvec = np.zeros(maxiteration) #Vector with the kktnorm for every iteration
    plot_rate = 10                       #HOW OFTEN PLOTS THE RESULTS

    # initialize variables for plot
    tt = np.arange(0, 2 * np.pi, 0.005)
    tt = np.kron((np.ones(Xc.shape[0])),tt[np.newaxis].T).T
    cc = np.cos(tt)
    ss = np.sin(tt)
    if 'kktnorm' == stopping_criteria:
        stop_cond = iteration < maxiteration and kktnorm > kkttol
    elif 'change' == stopping_criteria:
        stop_cond = iteration < maxiteration and change > changetol
    else:
        print("Stopping Criteria should be a valid entry: 'kktnorm','change'")
        sys.exit()
    
    xPhys=np.zeros((nelx,nely))

    # Initialize plot and plot the initial design  
    plt.ion()  
    fig = plt.figure()
    # Density Plot
    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Density Plot')
    im1 = ax1.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    ax1.axis('off')
    # Components Plot
    ax2 = fig.add_subplot(222)
    ax2.title.set_text('Components Plot')
    ax2.axis('off')
    ax2.set_xlim([min(Xx),max(Xx)])
    ax2.set_ylim([min(Yy),max(Yy)])
    ax2.set_aspect(aspect=1)
    #Objective Function PLot
    ax3 = fig.add_subplot(223)
    ax3.title.set_text('Objective Function')
    ax3.set_ylabel('c')
    ax3.set_xlabel('Iterations')
    ax3.set_ylim([0, 1000])
    #Volume Fraction Plot
    ax4 = fig.add_subplot(224,sharex = ax3)
    ax4.title.set_text('Volume Fraction')
    ax4.set_ylabel('v')
    ax4.set_xlabel('Iterations')
    ax4.set_ylim([0, volfrac*1.1])
    
    fig.show()

    print("Start iteration")
    while stop_cond:
        iteration=iteration + 1

        #Uses 1.6s
        ## Project component on DZ
        W,dW_dX,dW_dY,dW_dT,dW_dL,dW_dh=Wgp(ugp[:,0],ugp[:,1],Xg,p)

        #Uses 0.3s
        if np.logical_or(settings=='GGP', settings=='MMC'):
            delta=     (np.sum(np.reshape(      W[idgp]*gauss_weight,(int(len(gauss_weight)/Ngp**2),1,         Ngp**2),order='F'),axis=2)/               np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 )).T
            ddelta_dX= np.sum(np.reshape(dW_dX[:,idgp]*gauss_weight,(len(dW_dX),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dY= np.sum(np.reshape(dW_dY[:,idgp]*gauss_weight,(len(dW_dY),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dT= np.sum(np.reshape(dW_dT[:,idgp]*gauss_weight,(len(dW_dT),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dL= np.sum(np.reshape(dW_dL[:,idgp]*gauss_weight,(len(dW_dL),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dh= np.sum(np.reshape(dW_dh[:,idgp]*gauss_weight,(len(dW_dh),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            delta_c=   (np.sum(np.reshape(   W[idgp]**q*gauss_weight,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2)/np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 )).T
            ddelta_c_dX = np.sum(np.reshape(q*dW_dX[:,idgp] * W[idgp] ** (q - 1) * gauss_weight,(len(dW_dX),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dY = np.sum(np.reshape(q*dW_dY[:,idgp] * W[idgp] ** (q - 1) * gauss_weight,(len(dW_dY),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dT = np.sum(np.reshape(q*dW_dT[:,idgp] * W[idgp] ** (q - 1) * gauss_weight,(len(dW_dT),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dL = np.sum(np.reshape(q*dW_dL[:,idgp] * W[idgp] ** (q - 1) * gauss_weight,(len(dW_dL),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dh = np.sum(np.reshape(q*dW_dh[:,idgp] * W[idgp] ** (q - 1) * gauss_weight,(len(dW_dh),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
        elif np.logical_or(settings=='MNA', settings=='GP'):
            delta=     np.sum(np.reshape(W.T[idgp].T  *gauss_weight,   (W.shape[0],int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2) / np.sum(np.reshape(np.tile(gauss_weight,(W.shape[0],1)), (W.shape[0],int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2 )
            ddelta_dX= np.sum(np.reshape(dW_dX[:,idgp]*gauss_weight,(len(dW_dX),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dY= np.sum(np.reshape(dW_dY[:,idgp]*gauss_weight,(len(dW_dY),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dT= np.sum(np.reshape(dW_dT[:,idgp]*gauss_weight,(len(dW_dT),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dL= np.sum(np.reshape(dW_dL[:,idgp]*gauss_weight,(len(dW_dL),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_dh= np.sum(np.reshape(dW_dh[:,idgp]*gauss_weight,(len(dW_dh),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            delta_c=  np.sum(np.reshape(W.T[idgp].T**q*gauss_weight,   (W.shape[0],int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2) / np.sum(np.reshape(np.tile(gauss_weight,(W.shape[0],1)), (W.shape[0],int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2 )
            ddelta_c_dX = np.sum(np.reshape(q*dW_dX[:,idgp] * W.T[idgp].T ** (q - 1) * gauss_weight,(len(dW_dX),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dY = np.sum(np.reshape(q*dW_dY[:,idgp] * W.T[idgp].T ** (q - 1) * gauss_weight,(len(dW_dY),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dT = np.sum(np.reshape(q*dW_dT[:,idgp] * W.T[idgp].T ** (q - 1) * gauss_weight,(len(dW_dT),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dL = np.sum(np.reshape(q*dW_dL[:,idgp] * W.T[idgp].T ** (q - 1) * gauss_weight,(len(dW_dL),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
            ddelta_c_dh = np.sum(np.reshape(q*dW_dh[:,idgp] * W.T[idgp].T ** (q - 1) * gauss_weight,(len(dW_dh),int(len(gauss_weight)/Ngp**2),Ngp**2),order='F'),axis=2)  /    np.ravel(np.sum(np.reshape(gauss_weight[np.newaxis].T,(int(len(gauss_weight)/Ngp**2),1,Ngp**2),order='F'),axis=2 ))
        else:
            sys.exit()

        #Uses 0s
        ##Model Update
        E,dE,dE_dm=model_updateM(delta_c,p,X)
        rho,drho_ddelta,drho_dm=model_updateV(delta,p,X)
        dE_dX=dE * ddelta_c_dX
        dE_dY=dE * ddelta_c_dY
        dE_dT=dE * ddelta_c_dT
        dE_dL=dE * ddelta_c_dL
        dE_dh=dE * ddelta_c_dh
        drho_dX=drho_ddelta * ddelta_dX
        drho_dY=drho_ddelta * ddelta_dY
        drho_dT=drho_ddelta * ddelta_dT
        drho_dL=drho_ddelta * ddelta_dL
        drho_dh=drho_ddelta * ddelta_dh   
        rho[emptyelts]=0
        rho[fullelts]=1
        #E = E + 4 * E ** 3 - 7 * E ** 4 + 3 * E ** 5
        E[emptyelts]=p['Emin']
        E[fullelts]=p['E0']
        xPhys=rho
        E=np.reshape(np.ravel(E),(nelx,nely)).T

        # Uses 0.8s
        ### FE ANALYSIS
        sK = np.ravel(np.ravel(KE)[np.newaxis] * np.ravel(E.T)[np.newaxis].T)
        K = scipy.sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[freedofs, :][:, freedofs]
        U = np.zeros((ndof, 1))
        U[freedofs, 0] = scipy.sparse.linalg.spsolve(K, f[freedofs, 0])

        # Uses 0s
        # Objective function and sensitivity    
        ce = np.reshape((np.dot(U[edofMat].reshape(nelx * nely, 8), KE) * U[edofMat].reshape(nelx * nely, 8)).sum(1),(nely,nelx),order='F')
        c = (E * ce).sum()                      #Objective function
        v = np.mean(xPhys)                      #Volume fraction
        
        dc_dE= -np.ravel(ce,order='F')
        dc_dE[emptyelts]=0
        dc_dE[fullelts]=0
        dc_dX=np.dot(dE_dX, dc_dE[np.newaxis].T)
        dc_dY=np.dot(dE_dY, dc_dE[np.newaxis].T)
        dc_dL=np.dot(dE_dL, dc_dE[np.newaxis].T)
        dc_dh=np.dot(dE_dh, dc_dE[np.newaxis].T)
        dc_dT=np.dot(dE_dT, dc_dE[np.newaxis].T)
        dc_dm=np.dot(dE_dm, dc_dE[np.newaxis].T)
        dc=np.zeros(len(X))
        dc[np.arange(0,len(dc),6)]=np.ravel(dc_dX)
        dc[np.arange(1,len(dc),6)]=np.ravel(dc_dY)
        dc[np.arange(2,len(dc),6)]=np.ravel(dc_dL)
        dc[np.arange(3,len(dc),6)]=np.ravel(dc_dh)
        dc[np.arange(4,len(dc),6)]=np.ravel(dc_dT)
        dc[np.arange(5,len(dc),6)]=np.ravel(dc_dm)
        
        dv_dxPhys = np.ones(nely * nelx) / nelx / nely
        dv_dxPhys[emptyelts] = 0
        dv_dxPhys[fullelts] = 0
        dv_dX=np.dot(drho_dX, dv_dxPhys[np.newaxis].T)
        dv_dY=np.dot(drho_dY, dv_dxPhys[np.newaxis].T)
        dv_dL=np.dot(drho_dL, dv_dxPhys[np.newaxis].T)
        dv_dh=np.dot(drho_dh, dv_dxPhys[np.newaxis].T)
        dv_dT=np.dot(drho_dT, dv_dxPhys[np.newaxis].T)
        dv_dm=np.dot(drho_dm, dv_dxPhys[np.newaxis].T)
        dv=np.zeros(len(X))
        dv[np.arange(0,len(dv),6)]=np.ravel(dv_dX)
        dv[np.arange(1,len(dv),6)]=np.ravel(dv_dY)
        dv[np.arange(2,len(dv),6)]=np.ravel(dv_dL)
        dv[np.arange(3,len(dv),6)]=np.ravel(dv_dh)
        dv[np.arange(4,len(dv),6)]=np.ravel(dv_dT)
        dv[np.arange(5,len(dv),6)]=np.ravel(dv_dm)
        
        cvec[iteration-1]=c         #records objective function for every iteration
        vvec[iteration-1]=v         #records volume for every iteration
        kktnormvec[iteration-1]=kktnorm
        f0val=np.log(c + 1)
        fval=((v - volfrac) / volfrac) * 100
        df0dx=(np.ravel(dc) / (c + 1) * (np.ravel(upper_bound) - np.ravel(lower_bound)))
        dfdx=(np.ravel(dv).T / volfrac) * 100.0 * (np.ravel(upper_bound) - np.ravel(lower_bound)).T

        if 'Top_Rib' == BC:
            '''
            
            den?
            compare matlab den with my xPhys
            
        
            U1 = U(:,1) ; U2 = U(:,2) ;
            ce1 = sum((U1(edofMat)*KE).*U1(edofMat),2);
            ce2 = sum((U2(edofMat)*KE).*U2(edofMat),2);
            ce = ce1 + ce2 ;
            c = sum(sum((Emin+den.^penal*(E0-Emin)).*ce));
            dc = -penal*(E0-Emin)*grad_den'*(den.^(penal-1).*ce);
            cst=(sum(den(:))-mMax)/mMax*100;
            dcst=sum(grad_den)/mMax*100;
            dc=dc.*(Xmax-Xmin);dcst=dcst.*((Xmax-Xmin)');
            change=max(abs(X_old-X));
            X_old=X;
            f0val=c/10000;
            fval=cst;
            df0dx=dc(:)/10000;
            dfdx_=dcst(:)';
            '''



        if iteration == 1000:
            print(upp)
            print(upp.shape)

            aaaaa


        # output Vectors
        outvector1=[iteration,f0val,fval]
        outvector2=xval
    
        # Print iteration results
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, kktnorm.: {3:.3f}  ch.: {4:.3f}".format( iteration, c, v ,kktnorm, change))
        
        # Plot update
        if iteration % plot_rate == 0:
            
            ## Density plot
            im1.set_array(-xPhys.reshape((nelx, nely)).T)
            
            ## Component plot        
            Xc=Xg[np.arange(0,len(Xg),6)]
            Yc=Xg[np.arange(1,len(Xg),6)]
            Lc=Xg[np.arange(2,len(Xg),6)]
            hc=Xg[np.arange(3,len(Xg),6)]
            Tc=Xg[np.arange(4,len(Xg),6)]
            Mc=Xg[np.arange(5,len(Xg),6)]
            
            C0=np.outer(np.ones(len(cc[0])),np.cos(Tc)).T
            S0=np.outer(np.ones(len(cc[0])),np.sin(Tc)).T
            xxx=np.outer(np.ones(len(cc[0])),Xc).T+cc
            yyy=np.outer(np.ones(len(cc[0])),Yc).T+ss
            xi=C0 * (xxx-np.outer(np.ones(len(cc[0])),Xc).T) + S0 * (yyy-np.outer(np.ones(len(cc[0])),Yc).T)
            Eta=-S0 * (xxx-np.outer(np.ones(len(cc[0])),Xc).T) + C0 * (yyy-np.outer(np.ones(len(cc[0])),Yc).T)
            dd=np.real(norato_bar(xi,Eta,np.outer(np.ones(len(cc[0])),Lc).T,np.outer(np.ones(len(cc[0])),hc).T))
            xn=np.outer(np.ones(len(cc[0])),Xc).T+dd*cc
            yn=np.outer(np.ones(len(cc[0])),Yc).T+dd*ss
            
            tolshow=0.1
            Shown_compo=np.where(Mc > tolshow)
            Shown_compo=Shown_compo[0]

            ax2.fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],"w",edgecolor='black', linewidth=3)
            ax2.fill(xn[Shown_compo,:].T,yn[Shown_compo,:].T,Mc[Shown_compo].T,"b",edgecolor='blue', linewidth=1)
            if BC == 'L-Shape':
                ax2.fill([(min(Xx)+max(Xx))/2,max(Xx),max(Xx),(min(Xx)+max(Xx))/2],[(min(Yy)+max(Yy))/2,(min(Yy)+max(Yy))/2,max(Yy),max(Yy)],"w",edgecolor='black', linewidth=1)
            
            # Objective function plot
            ax3.plot(np.arange(1,iteration),cvec[np.arange(1,iteration)],'b', marker = 'o')
           
            # Volume fraction plot
            ax4.plot(np.arange(1,iteration),vvec[np.arange(1,iteration)],'r', marker = 'o')
            
            fig.canvas.draw()

        # Uses 1s
        ## MMA code optimization
        X,ymma,zmma,lam,xsi,eta,mu,zet,S,low,upp=mmasub(m,n,iteration,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,C,d)

        # Uses 0s
        xold2=xold1
        xold1=xval
        xval=X
        Xg=lower_bound + (upper_bound - lower_bound) * X
        change=np.linalg.norm((xval - xold1))               # Updates change value

        # Uses 0s
        ## The Residual Vector of the KKT conditions is calculated
        residu,kktnorm,residumax=kktcheck(m,n,X,ymma,zmma,lam,xsi,eta,mu,zet,S,xmin,xmax,df0dx,fval,dfdx,a0,a,C,d)
        #outvector1=[iteration,f0val,fval]
        #outvector2=xval
        
        # Updating stopping condition
        if 'kktnorm' == stopping_criteria:
            stop_cond=iteration < maxiteration and kktnorm > kkttol
        elif 'change' == stopping_criteria:
            stop_cond=iteration < maxiteration and change > changetol
    
    ## Density plot
    im1.set_array(-xPhys.reshape((nelx, nely)).T)
    
    ## Component plot        
    Xc=Xg[np.arange(0,len(Xg),6)]
    Yc=Xg[np.arange(1,len(Xg),6)]
    Lc=Xg[np.arange(2,len(Xg),6)]
    hc=Xg[np.arange(3,len(Xg),6)]
    Tc=Xg[np.arange(4,len(Xg),6)]
    Mc=Xg[np.arange(5,len(Xg),6)]
    
    C0=np.outer(np.ones(len(cc[0])),np.cos(Tc)).T
    S0=np.outer(np.ones(len(cc[0])),np.sin(Tc)).T
    xxx=np.outer(np.ones(len(cc[0])),Xc).T+cc
    yyy=np.outer(np.ones(len(cc[0])),Yc).T+ss
    xi=C0 * (xxx-np.outer(np.ones(len(cc[0])),Xc).T) + S0 * (yyy-np.outer(np.ones(len(cc[0])),Yc).T)
    Eta=-S0 * (xxx-np.outer(np.ones(len(cc[0])),Xc).T) + C0 * (yyy-np.outer(np.ones(len(cc[0])),Yc).T)
    dd=np.real(norato_bar(xi,Eta,np.outer(np.ones(len(cc[0])),Lc).T,np.outer(np.ones(len(cc[0])),hc).T))
    xn=np.outer(np.ones(len(cc[0])),Xc).T+dd*cc
    yn=np.outer(np.ones(len(cc[0])),Yc).T+dd*ss
    
    tolshow=0.1
    Shown_compo=np.where(Mc > tolshow)
    Shown_compo=Shown_compo[0]

    ax2.fill([min(Xx),max(Xx),max(Xx),min(Xx)],[min(Yy),min(Yy),max(Yy),max(Yy)],"w",edgecolor='black', linewidth=3)
    ax2.fill(xn[Shown_compo,:].T,yn[Shown_compo,:].T,Mc[Shown_compo].T,"b",edgecolor='blue', linewidth=1)
    if BC == 'L-Shape':
        ax2.fill([(min(Xx)+max(Xx))/2,max(Xx),max(Xx),(min(Xx)+max(Xx))/2],[(min(Yy)+max(Yy))/2,(min(Yy)+max(Yy))/2,max(Yy),max(Yy)],"w",edgecolor='black', linewidth=3)
    
    # Objective function plot
    ax3.plot(np.arange(1,iteration),cvec[np.arange(1,iteration)],'b', marker = 'o')
   
    # Volume fraction plot
    ax4.plot(np.arange(1,iteration),vvec[np.arange(1,iteration)],'r', marker = 'o')
    
    fig.canvas.draw()
    
    # Saving Plot
    fig.savefig('Plots.png')     
    print("Plots Saved" )
    
    return iteration, cvec, vvec, xPhys, xn[Shown_compo,:].T, yn[Shown_compo,:].T, Mc[Shown_compo].T, kktnormvec



def lgwt(N,a,b):
    # lgwt.m This script is for computing definite integrals using Legendre-Gauss Quadrature.Computes the Legendre - Gauss nodes and weights on an interval [a, b] with truncation order N
    # Suppose you have a continuous function f(x) which is defined on[a, b]  which you can evaluate at any x in [a, b].Simply evaluate itat all of the values contained in the x vector to obtain a vector f.Then compute the definite integral using sum(f. * w);
    # Written by Greg von Winckel - 02 / 25 / 2004
    N = N - 1
    N1 = N + 1
    N2 = N + 2
    xu = np.linspace(-1, 1, N1)
    xu = xu.transpose()

    # Initial guess
    y = np.cos((2 * np.arange(0,N+1).transpose()+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

    # Legendre-Gauss Vandermonde Matrix
    L=np.zeros((N1, N2))

    # Compute the zeros of the N + 1 Legendre Polynomial using the recursion relation and the Newton - Raphson method
    y0 = 2
    eps = 2.2204e-16
    Lp = 0

    # Iterate until new points are uniformly within epsilon of old points
    while max(abs(y - y0)) > eps:
        L[:, 0]=1
        L[:, 1]=y
        for k in np.arange(2,N1+1):
            L[:, k]=((2 * k-1 ) * y * L[:, k-1] - (k-1) * L[:, k -2] ) / k

        Lp = (N2) * (L[:, N1-1] - y * L[:, N2-1])/ (1 - y ** 2)
        y0 = y
        y = y0 - L[:, N2-1]/ Lp

    # Linear map from [-1, 1] to[a, b]
    x = (a * (1 - y) + b * (1 + y)) / 2
    # Compute the weights
    w = (b - a) / ((1 - y **2) * Lp ** 2) * (N2 / N1) ** 2
    return x, w

# Element stiffness matrix
def lk():
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    return KE

def Wgp(x, y, Xc, p):
    #  Evaluate characteristic function in each Gauss point
    ii = np.arange(0, len(x))
    X = Xc[np.arange(0, len(Xc), 6)]
    Y = Xc[np.arange(1, len(Xc), 6)]
    L = Xc[np.arange(2, len(Xc), 6)]
    h = Xc[np.arange(3, len(Xc), 6)]
    T = Xc[np.arange(4, len(Xc), 6)]
    jj = np.arange(0, len(X))
    I, J = np.meshgrid(ii, jj)

    # Uses 0.06s
    xi = np.kron(np.ones(len(I)),x[np.newaxis].T).T
    yi = np.kron(np.ones(len(I)),y[np.newaxis].T).T

    # Uses 0.4s
    rho = np.sqrt((X[J] - xi) ** 2 + (Y[J] - yi) ** 2)
    drho_dX = (X[J] - xi) / (rho + (rho == 0))
    drho_dY = (Y[J] - yi) / (rho + (rho == 0))
    phi = np.arctan2( -Y[J] + yi, - (X[J] - xi)) - T[J]
    dphi_dX = ((- Y[J] + yi) / (rho ** 2 + (rho == 0)))
    dphi_dY = (X[J] - xi) / (rho ** 2 + (rho == 0))
    dphi_dT = - np.ones((np.shape(J)))

    # Uses 0.7s
    upsi = np.sqrt(rho ** 2 + L[J] ** 2 / 4 - rho * L[J] * abs(np.cos(phi))) * (((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4)) + np.logical_not(((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4)) * abs(rho * np.sin(phi))
    dupsi_drho = (2 * rho - L[J] * abs(np.cos(phi))) / 2 / (upsi + (upsi == 0)) * ((((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4))) + np.logical_not(((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4)) * abs(np.sin(phi))
    dupsi_dphi = (L[J] * rho * np.sign(np.cos(phi)) * np.sin(phi)) / 2 / (upsi + (upsi == 0)) * ((((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4))) + np.logical_not((((rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4))) * rho * np.sign(np.sin(phi)) * np.cos(phi)
    dupsi_dL = (L[J] / 2 - rho * abs(np.cos(phi))) / 2 / (upsi + (upsi == 0)) * np.logical_and(((( rho * np.cos(phi)) ** 2) >= (L[J] ** 2 / 4)), upsi != 0)

    #Uses 0.5s
    if 'MMC' == p['method']:
        alp = p['alp']
        epsi = p['epsi']
        bet = p['bet']
        chi0 = 1 - (4 * upsi ** 2 / h[J] ** 2) ** alp
        dchi0_dh = 8 * alp * upsi ** 2.0 * (4 * upsi ** 2 / h[J] ** 2) ** (alp - 1) / h[np.newaxis].T ** 3
        dchi0_dupsi = -8 * alp * upsi * (4 * upsi ** 2.0 / h[J] ** 2) ** (alp - 1) / h[np.newaxis].T ** 2
        chi, dchi = Aggregation_Pi(chi0, p)
        dchi_dh = (dchi0_dh * dchi)
        dchi_dupsi = (dchi0_dupsi * dchi)
        chi[chi <= - 1000000.0] = - 1000000.0
        W = (chi > epsi) + np.logical_and(chi <= epsi, chi>= - epsi) * (3 / 4 * (1 - bet) * (chi / epsi - chi ** 3 / 3 / epsi ** 3) + (1 + bet) / 2) + (chi < - epsi) * bet
        dW_dchi = - 3 / 4 * (1 / epsi - chi ** 2 / epsi ** 3) * (bet - 1) * (abs(chi) < epsi)
        dW_dupsi = np.outer(np.ones(len(dchi_dh)), dW_dchi) * dchi_dupsi
        dW_dh = np.outer(np.ones(len(dchi_dh)), dW_dchi)* dchi_dh
        dW_dX = dW_dupsi * (dupsi_dphi * dphi_dX + dupsi_drho * drho_dX)
        dW_dY = dW_dupsi * (dupsi_dphi * dphi_dY + dupsi_drho * drho_dY)
        dW_dL = dW_dupsi * dupsi_dL
        dW_dT = dW_dupsi * dupsi_dphi * dphi_dT
    elif 'MNA' == p['method']:
        epsi = p['sigma']
        ds = upsi
        d = abs(upsi)
        l = h[J] / 2 - epsi / 2
        u = h[J] / 2 + epsi / 2
        a3 = - 2.0 / ((l - u) * (l ** 2 - 2 * l * u + u ** 2))
        a2 = (3 * (l + u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2))
        a1 = - (6 * l * u) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2))
        a0 = (u * (- u ** 2 + 3 * l * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2))
        W = 1 * (d <= l) + (a3 * d ** 3 + a2 * d ** 2 + a1 * d + a0) * np.logical_and(d <= u, d > l)
        dW_dupsi = np.sign(ds) * (3 * a3 * d ** 2 + 2 * a2 * d + a1) * np.logical_and(d <= u, d > l)
        da3_du = -2 / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) - (2 * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        da2_du = 3 / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) + (3 * (l + u)) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) + (3 * (l + u) * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        da1_du = - (6 * l) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) - (6 * l * u) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) - (6 * l * u * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        da0_du = (- u ** 2 + 3 * l * u) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) + (u * (- u ** 2 + 3 * l * u)) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) + (u * (3 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) + (u * (- u ** 2 + 3 * l * u) * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        dWf_du = (da3_du * d** 3 + da2_du * d ** 2 + da1_du * d + da0_du) * np.logical_and(d <= u, d > l)
        da3_dl = 2 / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) + (2 * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        da2_dl = 3 / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) - (3 * (l + u)) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) - (3 * (l + u) * (2 * l - 2 * u))/ ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        da1_dl = (6 * l * u) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) - (6 * u) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) + (6 * l * u * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)** 2)
        da0_dl = (3 * u ** 2) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2)) - (u * (- u ** 2 + 3 * l * u)) / ((l - u) ** 2 * (l ** 2 - 2 * l * u + u ** 2)) - (u * (- u ** 2 + 3 * l * u) * (2 * l - 2 * u)) / ((l - u) * (l ** 2 - 2 * l * u + u ** 2) ** 2)
        dWf_dl = (da3_dl * d ** 3 + da2_dl * d ** 2 + da1_dl * d + da0_dl) * np.logical_and(d <= u, d > l)
        dW_dh = 0.5 * np.sign(ds) * (dWf_du + dWf_dl)
        dW_dX = dW_dupsi * (dupsi_dphi * dphi_dX + dupsi_drho * drho_dX)
        dW_dY = dW_dupsi * (dupsi_dphi * dphi_dY + dupsi_drho * drho_dY)
        dW_dL = dW_dupsi * dupsi_dL
        dW_dT = dW_dupsi * dupsi_dphi * dphi_dT
    elif 'GP' == p['method']:
        deltamin = p['deltamin']
        r = p['r']
        zetavar = upsi - h[J] / 2
        dzetavar_dupsi = np.ones(upsi.shape)
        dzetavar_dh = -0.5 * np.ones(J.shape)
        deltaiel = (1 / np.pi / r ** 2 * (r ** 2 * scimath.arccos(zetavar / r) - zetavar * scimath.sqrt(r ** 2 - zetavar ** 2))) * (abs(zetavar) <= r) + (zetavar < - r)
        deltaiel = deltaiel.real
        ddetlaiel_dzetavar = (- 2 * scimath.sqrt(r ** 2 - zetavar ** 2) / np.pi / r ** 2) * (abs(zetavar) <= r)
        ddetlaiel_dzetavar = ddetlaiel_dzetavar.real
        W = deltamin + (1 - deltamin) * deltaiel
        dW_ddeltaiel = (1 - deltamin)
        dW_dh = dW_ddeltaiel * ddetlaiel_dzetavar * dzetavar_dh
        dW_dupsi = dW_ddeltaiel * ddetlaiel_dzetavar * dzetavar_dupsi
        dW_dX = dW_dupsi * (dupsi_dphi * dphi_dX + dupsi_drho * drho_dX)
        dW_dY = dW_dupsi * (dupsi_dphi * dphi_dY + dupsi_drho * drho_dY)
        dW_dL = dW_dupsi * dupsi_dL
        dW_dT = dW_dupsi * dupsi_dphi * dphi_dT
    else:
        sys.exit()
        
    return W, dW_dX, dW_dY, dW_dT, dW_dL, dW_dh

def Aggregation_Pi(z, p):
    # function that make the aggregation of the value z and also compute sensitivities
    zm = np.kron(np.ones(len(z)), np.amax(z, axis=0)[np.newaxis].T).T
    ka = p['ka']
    if 'p-norm' == p['aggregation']:
        zp = p['zp']
        zm = zm + zp
        z = z + zp
        Wa = np.exp(zm[1, :]) * (sum((z / np.exp(zm)) ** ka, 0)) ** (1 / ka) - zp
        dWa = (z / np.exp(zm)) ** (ka - 1) * np.kron(np.ones(len(z)),((sum((z / np.exp(zm)) ** ka, 0)) ** (1 / ka - 1))[np.newaxis].T).T
    elif 'p-mean' == p['aggregation']:
        zp = p['zp']
        zm = zm + zp
        z = z + zp
        Wa = np.exp(zm[1, :]) * ((np.mean(((z / np.exp(zm)) ** ka), axis=0)) ** (1 / ka)) - zp
        dWa = 1 / len(z) ** (1 / ka) * (z / np.exp(zm)) ** (ka - 1) * np.kron(np.ones(len(z)), ((sum((z / np.exp(zm)) ** ka, 0)) ** (1 / ka - 1))[np.newaxis].T).T
    elif 'KS' == p['aggregation']:
        Wa = zm[1, :] + 1 / ka * np.log(sum(np.exp(ka * (z - zm)), 0))
        dWa = np.exp(ka * (z - zm)) / np.outer(np.ones(len(z)), sum(np.exp(ka * (z - zm)), 0))
    elif 'KSl' == p['aggregation']:
        Wa = zm[1, :] + 1 / ka * np.log(np.mean(np.exp(ka * (z - zm)), axis=0))
        dWa = np.exp(ka * (z - zm)) / np.outer(np.ones(len(z)),sum(np.exp(ka * (z - zm)),0))
    elif 'IE' == p['aggregation']:
        Wa = sum(z * np.exp(ka * (z - zm))) / sum(np.exp(ka * (z - zm)), 0)
        dWa = ((np.exp(ka * (z - zm)) + ka * z * np.exp(ka * (z - zm))) * np.outer(np.ones(len(z)),sum(np.exp(ka * (z - zm)),0)) - np.outer(np.ones(len(z)),sum(z * np.exp(ka * (z - zm)),0)) * ka * np.exp(ka * (z - zm))) / np.outer(np.ones(len(z)), sum(np.exp(ka * (z - zm)), 0) ** 2)
    else:
        sys.exit()
    return Wa, dWa

def model_updateM(delta,p,X):
    m=X[np.arange(5,len(X),6)]
    nc=len(m)
    m=np.outer(np.ones(delta.shape[1]),m).T
    if 'MMC' == p['method']:
        #update the Young Modulus on the base of delta
        E=p['E0'] * delta
        dE_ddelta=p['E0'] * np.ones(delta.shape[1])
        dE_ddelta=np.outer(np.ones(m.shape[0]),dE_ddelta)
        dE_dm=0 * m
    elif 'MNA' == p['method']:
        hatdelta=np.multiply(delta,m ** p['gammac'])
        rho,drho_dhatdelta = Aggregation_Pi(hatdelta,p)
        if p['saturation']:
            rho,ds = smooth_sat(rho,p,nc)
            drho_dhatdelta=ds * drho_dhatdelta
        E=rho ** p['penalty'] * (p['E0'] - p['Emin']) + p['Emin']
        dhatdelta_ddelta=m ** p['gammac']
        dhatdelta_dm = p['gammac'] * np.multiply(delta, m ** (p['gammac'] - 1))
        dE_ddelta=p['penalty'] *(p['E0'] - p['Emin']) * dhatdelta_ddelta * drho_dhatdelta * rho ** (p['penalty'] - 1)
        dE_dm=p['penalty'] * (p['E0'] - p['Emin']) * dhatdelta_dm * drho_dhatdelta * rho ** (p['penalty'] - 1)
    elif 'GP' == p['method']:
        hatdelta= np.multiply(delta,m ** p['gammac'])
        E,dE_dhatdelta = Aggregation_Pi(hatdelta,p)
        if p['saturation']:
            E,ds=smooth_sat(E,p,nc)
            dE_dhatdelta=ds * dE_dhatdelta
        E=E * p['E0']
        dhatdelta_ddelta=m ** p['gammac']
        dhatdelta_dm=p['gammac'] * np.multiply(delta,m ** (p['gammac'] - 1))
        dE_ddelta=p['E0'] * dhatdelta_ddelta * dE_dhatdelta
        dE_dm=p['E0'] * dE_dhatdelta * dhatdelta_dm
    else:
        sys.exit()
    return E,dE_ddelta,dE_dm

def model_updateV(delta,p,X):
    m = X[np.arange(5, len(X), 6)]
    nc = len(m)
    m = np.outer(np.ones(delta.shape[1]), m).T
    if 'MMC' == p['method']:
        #update the Young Modulus on the base of delta
        rho=delta
        drho_ddelta=np.ones(delta.shape[1])
        drho_ddelta=np.outer(np.ones(m.shape[0]),drho_ddelta)
        drho_dm=0 * m
    elif 'MNA' == p['method']:
        hatdelta = np.multiply(delta, m ** p['gammav'])
        rho,drho_dhatdelta = Aggregation_Pi(hatdelta,p)
        if p['saturation']:
            rho,ds = smooth_sat(rho,p,nc)
            drho_dhatdelta=ds * drho_dhatdelta
        dhatdelta_ddelta=m ** p['gammav']
        dhatdelta_dm = p['gammav'] * np.multiply(delta, m ** (p['gammav'] - 1))
        drho_ddelta=dhatdelta_ddelta * drho_dhatdelta
        drho_dm=drho_dhatdelta * dhatdelta_dm
    elif 'GP' == p['method']:
        hatdelta = np.multiply(delta, m ** p['gammav'])
        rho,drho_dhatdelta = Aggregation_Pi(hatdelta,p)
        if p['saturation']:
            rho,ds = smooth_sat(rho,p,nc)
            drho_dhatdelta=ds * drho_dhatdelta
        dhatdelta_ddelta = m ** p['gammav']
        dhatdelta_dm = p['gammav'] * np.multiply(delta, m ** (p['gammav'] - 1))
        drho_ddelta=dhatdelta_ddelta * drho_dhatdelta
        drho_dm=drho_dhatdelta * dhatdelta_dm
    else:
        sys.exit()
    return rho,drho_ddelta,drho_dm

def smooth_sat(y, p, nc):
    if 'p-norm' == p['aggregation']:
        xt = 1
    elif 'p-mean' == p['aggregation']:
        xt = (((nc - 1) *  p['zp'] ** p['ka'] + (1 + p['zp']) ** p['ka']) / nc) ** (1 / p['ka']) - p['zp']
    elif 'KS' == p['aggregation']:
        xt = 1
    elif 'KSl' == p['aggregation']:
        xt = 1 + 1 / p['ka'] * np.log((1 + (nc - 1) * np.exp(- p['ka'])) / nc)
    elif 'IE' == p['aggregation']:
        xt = 1 + 1 / p['ka'] * np.log((1 + (nc - 1) * np.exp(- p['ka'])) / nc)
    else:
        sys.exit()

    pp = 100
    s0 = - np.log(np.exp(- pp) + 1.0 / (np.exp((pp * 0) / xt) + 1.0)) / pp
    s = lambda xs=None, a=None, pa=None: (- np.log(np.exp(- pa) + 1.0 / (np.exp((pa * xs) / a) + 1.0)) / pa - s0) / (1 - s0)
    ds = lambda xs=None, a=None, pa=None: (np.exp((pa * xs) / a * 1.0) / (np.exp((pa * xs) / a) + 1.0) ** 2) / (a * (np.exp(- pa) + 1.0 / (np.exp((pa * xs) / a) + 1.0))) / (1 - s0)
    # syms a xs
    s = s(y, xt, pp)
    ds = ds(y, xt, pp)
    return s, ds

def norato_bar(xi, eta, L, h):
    
    d=((L/2* scimath.sqrt(xi**2/(xi**2+eta**2))+scimath.sqrt(h**2/4-L**2/4*eta**2/(xi**2+eta**2)))*(xi**2/(xi**2+eta**2)>=(L**2/(h**2+L**2)))+h/2*scimath.sqrt(1+xi**2/(eta**2+(eta==0))) * (xi**2/(xi**2+eta**2)<(L**2/(h**2+L**2))))*np.logical_or(xi!=0,eta!=0)+scimath.sqrt(2)/2*h*np.logical_and(xi==0,eta==0)
    return d

def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    #    Version September 2007 (and a small change August 2008)
    #    Krister Svanberg <krille@math.kth.se>
    #    Department of Mathematics, SE-10044 Stockholm, Sweden.
    
    #    This function mmasub performs one MMA-iteration, aimed at
    #    solving the nonlinear programming problem:
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmin_j <= x_j <= xmax_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    # *** INPUT:
    #   m    = The number of general constraints.
    #   n    = The number of variables x_j.
    #  iter  = Current iteration number ( =1 the first time mmasub is called).
    #  xval  = Column vector with the current values of the variables x_j.
    #  xmin  = Column vector with the lower bounds for the variables x_j.
    #  xmax  = Column vector with the upper bounds for the variables x_j.
    #  xold1 = xval, one iteration ago (provided that iter>1).
    #  xold2 = xval, two iterations ago (provided that iter>2).
    #  f0val = The value of the objective function f_0 at xval.
    #  df0dx = Column vector with the derivatives of the objective function
    #          f_0 with respect to the variables x_j, calculated at xval.
    #  fval  = Column vector with the values of the constraint functions f_i,
    #          calculated at xval.
    #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at xval.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #  low   = Column vector with the lower asymptotes from the previous
    #          iteration (provided that iter>1).
    #  upp   = Column vector with the upper asymptotes from the previous
    #          iteration (provided that iter>1).
    #  a0    = The constants a_0 in the term a_0*z.
    #  a     = Column vector with the constants a_i in the terms a_i*z.
    #  c     = Column vector with the constants c_i in the terms c_i*y_i.
    #  d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    # *** OUTPUT:
    #  xmma  = Column vector with the optimal values of the variables x_j
    #          in the current MMA subproblem.
    #  ymma  = Column vector with the optimal values of the variables y_i
    #          in the current MMA subproblem.
    #  zmma  = Scalar with the optimal value of the variable z
    #          in the current MMA subproblem.
    #  lam   = Lagrange multipliers for the m general MMA constraints.
    #  xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general MMA constraints.
    #  low   = Column vector with the lower asymptotes, calculated and used
    #          in the current MMA subproblem.
    #  upp   = Column vector with the upper asymptotes, calculated and used
    #          in the current MMA subproblem.

    epsimin = 10 ** (- 7)
    raa0 = 1e-05
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.4
    
    # Calculation of the asymptotes low and upp :
    if iter < 2.5:
        move = 1
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        move = 0.5
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[np.where(zzz > 0)] = asyincr
        factor[np.where(zzz < 0)] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.1 * (xmax - xmin)
        lowmax = xval - 0.0001 * (xmax - xmin)
        uppmin = xval + 0.0001 * (xmax - xmin)
        uppmax = xval + 0.1 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)
        
    # Calculation of the bounds alfa and beta :
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)
    # Calculations of p0, q0, P, Q and b.
    xmami = xmax - xmin
    xmamieps = np.ones(n) * 1e-05
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = np.ones(n) / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = np.ones(n) / ux1
    xlinv = np.ones(n) / xl1    

    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(- df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(- dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.ones(m) * xmamiinv.T
    P = P + PQ
    Q = Q + PQ
    P = P * scipy.sparse.spdiags(ux2, 0, n, n)
    Q = Q * scipy.sparse.spdiags(xl2, 0, n, n) 
    b = np.dot(P[np.newaxis],uxinv[np.newaxis].T)[0] + np.dot(Q[np.newaxis],xlinv[np.newaxis].T)[0] - fval
    
    ## Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b,c, d)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp

def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    # This function subsolv solves the MMA subproblem:
    # minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
    #          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    # subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
    #            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.

    epsi = 1
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1
    lam = np.ones(m)
    xsi = np.ones(n) / (x - alfa)
    xsi = np.maximum(xsi, np.ones(n))
    eta = np.ones(n) / (beta - x)
    eta = np.maximum(eta, np.ones(n))
    mu = np.maximum(np.ones(m), 0.5 * c)
    zet = 1
    s = np.ones(m)

    it1=0
    while epsi > epsimin:
        it1 = it1 + 1
        
        epsvecn = epsi * np.ones(n)
        epsvecm = epsi * np.ones(m)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = np.ones(n) / ux1        
        xlinv1 = np.ones(n) / xl1
        plam = p0 + P.T * lam
        qlam = q0 + Q.T * lam
        gvec = np.dot(P[np.newaxis],uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis],xlinv1[np.newaxis].T)[0]
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta 
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T * lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate([rex, rey, rez]).T
        residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
        residu = np.concatenate([residu1, residu2])
        residunorm = np.sqrt(np.dot(residu[np.newaxis],residu[np.newaxis].T)[0][0])   #PROBABLY WRONH; THE VALUE IS TO HIGH WHEN COMPARED TO MATLAB
        residumax = max(abs(residu))
        
        it2 = 0
        while np.logical_and(residumax > 0.9 * epsi, it2 < 500):
            it2 = it2 + 1
            
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = np.ones(n) / ux1
            xlinv1 = np.ones(n) / xl1
            uxinv2 = np.ones(n) / ux2
            xlinv2 = np.ones(n) / xl2
            plam = p0 + P.T * lam
            qlam = q0 + Q.T * lam
            gvec = np.dot(P[np.newaxis],uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis],xlinv1[np.newaxis].T)[0]
            GG = P * scipy.sparse.spdiags(uxinv2, 0, n, n) - Q * scipy.sparse.spdiags(xlinv2, 0, n, n)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T * lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam            
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = np.ones(n) / diagx
            diagy = d + mu / y
            diagyinv = np.ones(m) / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv      
            if m > n:
                blam = dellam + dely / diagy - np.dot(GG[np.newaxis], (delx / diagx)[np.newaxis].T)[0]
                bb = np.concatenate([blam.T, delz]).T
                Alam = diaglamyi[0] + np.dot((GG * scipy.sparse.spdiags(diagxinv, 0, n, n))[np.newaxis], GG[np.newaxis].T)[0][0]
                AA = [[Alam, a[0]],[a[0], - zet / z]]                
                solut = np.linalg.solve(AA, bb)
                dlam = solut[np.arange(0, m)]
                dz = solut[m]
                dx = - delx / diagx - (GG.T * dlam) / diagx
            else:
                diaglamyiinv = np.ones(m) / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = scipy.sparse.spdiags(diagx,0,n,n) + GG[np.newaxis].T*scipy.sparse.spdiags(diaglamyiinv,0,m,m)*GG
                azz = zet / z + a.T * (a / diaglamyi)
                axz = - GG.T * (a / diaglamyi)
                bx = delx + GG.T * (dellamyi / diaglamyi)
                bz = delz - a.T* (dellamyi / diaglamyi)
                AA1 = np.c_[Axx, axz]
                AA2 = np.concatenate([ axz.T, azz])
                AA = np.r_[AA1, AA2[np.newaxis]]
                bb = np.concatenate([- bx.T, - bz])
                solut = np.linalg.solve(AA, bb)
                dx = solut[np.arange(0, n)]
                dz = solut[n]
                dlam = (np.dot(GG[np.newaxis], dx[np.newaxis].T)[0][0]) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi
            dy = - dely / diagy + dlam / diagy
            dxsi = - xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = - eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = - mu + epsvecm / y - (mu * dy) / y
            dzet = - zet + epsi / z - zet * dz / z
            ds = - s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y.T, [z], lam, xsi.T, eta.T, mu.T, [zet], s.T),axis = 0)
            dxx = np.concatenate((dy.T, [dz], dlam.T, dxsi.T, deta.T, dmu.T, [dzet], ds.T),axis = 0)
            stepxx = - 1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = - 1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1)
            steg = 1 / stminv
            xold = x
            yold = y
            zold = z
            lamold = lam
            xsiold = xsi
            etaold = eta
            muold = mu
            zetold = zet
            sold = s
            resinew = 2 * residunorm
            
            it3 = 0
            while np.logical_and(resinew > residunorm, it3 < 50).any():     
                it3 = it3 + 1
                
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = np.ones(n) / ux1
                xlinv1 = np.ones(n) / xl1
                plam = p0 + P.T * lam
                qlam = q0 + Q.T * lam
                gvec = np.dot(P[np.newaxis],uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis],xlinv1[np.newaxis].T)[0]
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T * lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate([rex.T, rey.T, rez]).T
                residu2 = np.concatenate([relam.T, rexsi.T, reeta.T, remu.T, [rezet], res.T]).T
                residu = np.concatenate([residu1.T, residu2.T]).T
                resinew = np.sqrt(np.dot(residu[np.newaxis],residu[np.newaxis].T)[0][0])
                steg = steg / 2  
            residunorm = resinew
            residumax = max(abs(residu))
            #steg = 2 * steg
        epsi = 0.1 * epsi
    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s   

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma

def kktcheck(m, n, x, y, z, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d):
    #  The left hand sides of the KKT conditions for the following
    #  nonlinear programming problem are calculated.
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmax_j <= x_j <= xmin_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    # *** INPUT:
    #   m    = The number of general constraints.
    #   n    = The number of variables x_j.
    #   x    = Current values of the n variables x_j.
    #   y    = Current values of the m variables y_i.
    #   z    = Current value of the single variable z.
    #  lam   = Lagrange multipliers for the m general constraints.
    #  xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general constraints.
    #  xmin  = Lower bounds for the variables x_j.
    #  xmax  = Upper bounds for the variables x_j.
    #  df0dx = Vector with the derivatives of the objective function f_0
    #          with respect to the variables x_j, calculated at x.
    #  fval  = Vector with the values of the constraint functions f_i,
    #          calculated at x.
    #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at x.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #   a0   = The constants a_0 in the term a_0*z.
    #   a    = Vector with the constants a_i in the terms a_i*z.
    #   c    = Vector with the constants c_i in the terms c_i*y_i.
    #   d    = Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    # *** OUTPUT:
    # residu     = the residual vector for the KKT conditions.
    # residunorm = sqrt(residu'*residu).
    # residumax  = max(abs(residu)).

    rex = df0dx + dfdx.T * lam - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - a.T * lam
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s
    residu1 = np.concatenate([rex.T, rey.T, rez]).T    
    residu2 = np.concatenate([relam.T, rexsi.T, reeta.T, remu.T, [rezet], res.T]).T
    residu = np.concatenate([residu1.T, residu2.T]).T
    residunorm = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T))[0][0]
    residumax = max(abs(residu))  

    return residu, residunorm, residumax

# The real main driver
if __name__ == "__main__":
    # Input parameters
    nelx=80
    nely=40
    volfrac=0.4
    settings='GGP'          #Method:   'GGP'   'MMC'   'MNA'   'GP'
    BC='Short_Cantiliever'                #Boundary Conditions:  'MBB'  'Short_Cantiliever'  'L-Shape'  'Top_Rib'
    maxiteration=300
    if len(sys.argv)>1: nelx = int(sys.argv[1])
    if len(sys.argv)>2: nely = int(sys.argv[2])
    if len(sys.argv)>3: volfrac = float(sys.argv[3])
    if len(sys.argv)>4: settings = str(sys.argv[4])
    if len(sys.argv)>5: BC = str(sys.argv[5])
    if len(sys.argv)>6: maxiteration = int(sys.argv[6])
    main(nelx,nely,volfrac,settings,BC,maxiteration)
