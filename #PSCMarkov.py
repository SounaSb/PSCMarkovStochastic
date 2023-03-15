## PSCMarkov

import numpy as np
import matplotlib.pyplot as plt
import math as mt
import copy
from tqdm import tqdm


## Le code qui calcul
def simul(N, M, D, R, T0, scen):
    U0, V0 = scen()
    T = np.zeros(N)
    U = U0.copy()
    V = V0.copy()
    T[0] = T0
    
    exp = np.random.exponential(1,N)
    bin = np.random.binomial(1, 0.5, N)
    plage = np.random.randint(1, 1+ M//40, N)
    
    # Optimisation
    evol = [(2,0,0)]
    mem_u = []
    mem_v = []
    mem_e = []

    dT = 0
    
    
    
    def type_ofjump(par_d,par_n,par_m):
        param = par_d + par_n + par_m
        
        pass
        
    
    def one_jump(u,v,mem_e,mem_u,mem_v,evol):
        pass

    def 
    
    
    

    for i in tqdm(range(1,N)):

        if len(mem_u) == 0 or len(mem_v) == 0 or len(mem_e) == 0:
    ### Paramètres de poids de chaque slot en fonction des population et du role (croisé, diff)
            ## Diffusion
            seg_ud = U*(D[0][0] + D[0][1]*U + D[0][2]*V) 
            seg_vd = V*(D[1][0] + D[1][1]*U + D[1][2]*V)
            ## Naissance
            seg_un = U*R[0][0] 
            seg_vn = V*R[1][0]
            ## Mort
            seg_um = U*(R[0][1]*U + R[0][2]*V) 
            seg_vm = V*(R[1][1]*U + R[1][2]*V)

            ## Paramètre de chaque action possible
            par_d = seg_ud.sum() + seg_vd.sum() # 
            par_n = seg_un.sum() + seg_vn.sum()
            par_m = seg_um.sum() + seg_vm.sum()
            param = par_d + par_n + par_m
            
           
            dT = exp[i]/param
            T[i] = T[i-1]+dT

    ### Qui saute
            ## Diffusion
            q = lambda_v / param
            mem_e = np.random.binomial(1, q, size=2000)
            mem_u = np.random.choice(range(len(seg_u)), p=seg_u/lambda_u, size=int((1-q)*2000)) 
            mem_v = np.random.choice(range(len(seg_v)), p=seg_v/lambda_v, size=int(q*2000)) 

    ### Actualisation
        ## Diffusion 
        e, mem_e = mem_e[-1], mem_e[: -1]
        if e == 0:
            p, mem_u = mem_u[-1], mem_u[: -1]
        else:
            p, mem_v = mem_v[-1], mem_v[: -1]

          
    ### Plus qu'à actualiser les positions avec le saut determiné
        dir = bin[i]
        k = int((p+plage[i]*(1-2*dir))%M)
        
        ## Diffusion
        if e == 0:
            if U[p]> 0 :
                U[k] += 1
                U[p] += -1
        else:
            if V[p] > 0:
                V[k] += 1
                V[p] += -1

        evol.append((e, k, p))
        

        ## On rajoute la partie naissance mort
  
    return evol, T







#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Partie animation
import matplotlib.animation as animation
import time
t = None
n=0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), mode='same')/w

def Animate(evol,scenario,T,vitesse,suivi,moy,filename=''): 
    t=np.linspace(0,T[-1],3*N)
    xmin = 0
    xmax = 1
    nbx = M
    dt = vitesse 

    U, V = scenario()
    Uprime = [U]
    Vprime = [V]
    Utemp, Vtemp = U.copy(), V.copy()

    Ptemp = [M//2]
    Pprime = [M//2]
    test = np.random.uniform(0,1, size=N)

    # On reconstruit notre évolution
    for i in tqdm(range(N)):
        
        # Construction de nos populations
        de, k, dp = evol[i]
        if de==0:
            Utemp[k] += 1
            Utemp[dp] += -1
        elif de == 1:
            Vtemp[k] += 1
            Vtemp[dp] += -1

        # Suivi position
        if suivi == evol[i][0]:
            if evol[i][2] == Ptemp:
                parc = [Utemp, Vtemp]
                if parc[suivi][Ptemp] == 0:
                    pass
                elif test[i]>1/parc[suivi][Ptemp]:
                    Ptemp = evol[i][1]

        # Pocessus d'acceleration
        if i % int(dt*M) == 0:
            Uprime.append(Utemp.copy())
            Vprime.append(Vtemp.copy())
            Pprime.append(Ptemp)
    
    x = np.linspace(xmin, xmax, nbx)

    fig = plt.figure()
    Uline, = plt.plot([], [], color ='red') 
    Vline, = plt.plot([], [], color ='blue')
    Pline, = plt.plot([], [], color ='green', marker='o', markersize=5) 

    plt.xlim(xmin, xmax)
    plt.ylim(-1, 1.5*np.max([np.max(U), np.max(V)]))

    st = plt.suptitle("", fontweight="bold")

    def anim(i):
        # Cas moyenné
        if moy:
            Uline.set_data(x, moving_average(Uprime[i],M//15))
            Vline.set_data(x, moving_average(Vprime[i],M//15))
            if suivi==0:
                Pline.set_data([Pprime[i]], [moving_average(Uprime[i],M//17)[Pprime[i]]])
            else:
                Pline.set_data([Pprime[i]], [moving_average(Vprime[i],M//17)(Vprime[i],M//17)[Pprime[i]]])

        # Avec bruit
        else:
            Uline.set_data(x, Uprime[i])
            Vline.set_data(x, Vprime[i])
            if suivi==0:
                Pline.set_data([Pprime[i]], [Uprime[i][Pprime[i]]])
            else:
                Pline.set_data([Pprime[i]], [Vprime[i][Pprime[i]]])
        
        st.set_text(str(int(100*(i/(N//(dt*M)))))+"%")

        return Uline, Vline, Pline
 
    ani = animation.FuncAnimation(fig, anim, frames= len(Uprime),interval=1, repeat=True)
    plt.show()
    
    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------





## On exécute maintenant
D = [[1, 1, 1] , [1, 1, 1]]
R = [[0, 0, 0] , [0, 0, 0]]
M = 100

N = 1000000
T0 = 0
absc = np.linspace(start = 0, stop = 1, num = M)

# Creneau 1/l nb de site au milieu
def creneau(x, M, prop, pos):
    k = M//prop
    for i in range(k):
        if pos == 'centre':
            x[i+int((prop-1)/2*k)] += 150
        elif pos == 'gauche':
            x[i+int((prop-1)/4*k)] += 150
        else:
            x[i+int(3*(prop-1)/4*k)] += 150


## Scenarios
# Intru scenario
def intru():
    U0 = 0*absc+80
    V0 = 0*absc+20
    creneau(V0, M, 6, 'centre')
    return U0, V0

# Deux creneaux décalés scenario
def crendec():
    U0 = 0*absc+30
    V0 = 0*absc+30
    creneau(V0, M, 10, 'gauche')
    creneau(U0, M, 10, 'droite')
    return U0, V0

# Formation X
def xlin():
    U0 = 200*absc+10
    V0 = 200*(1-absc)+10
    return U0, V0



# On calcul
evol,T = simul(N,M,D,T0,intru)

# On anime
Animate(evol, intru, T, vitesse = 2000/M, suivi=0, moy=False)

#Affichage des conditions initiales
#U0, V0 = intru()
#plt.plot(absc,U0, color = 'red') 
#plt.plot(absc,V0, color = 'blue')
#plt.show()