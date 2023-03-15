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
    bin = np.random.binomial(1, 0.5, N) # évolution à droite ou à gauche
    plage = np.random.randint(1, 1+ M//40, N)
    
    # Optimisation
    evol = [(2,0,0)]
    mem_u = []
    mem_v = []
    mem_e = []

    dT = 0
    
    
    
    def type_of_evol(par_d,par_n,par_m):
        total_weight = par_d + par_n + par_m
        return np.random.choice([0,1,2],
                                [par_d/total_weight,par_n/total_weight,par_m/total_weight])
        
    
    def one_jump(u,v,mem_e,mem_u,mem_v,evol):
        evol_type=type_of_evol
    
    
    

    for i in tqdm(range(1,N)):

        if len(mem_u) == 0 or len(mem_v) == 0 or len(mem_e) == 0: # on réactualise si une des listes mémoires est vide
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
            par_d = seg_ud.sum() + seg_vd.sum()
            par_n = seg_un.sum() + seg_vn.sum()
            par_m = seg_um.sum() + seg_vm.sum()
            param = par_d + par_n + par_m # somme de tous les poids : diffusion, naissance, mort et pour les 2 populations
            
           
            dT = exp[i]/param # intervalle de temps jusqu'au prochain saut
            T[i] = T[i-1]+dT # temps du saut i

            ### Qui saute
            ## Diffusion
            qd = seg_vd.sum() / par_d     # probabilité que ce soit v qui diffuse plutôt que u
            qn = seg_vn.sum() / par_n     # probabilité que ce soit v qui naisse plutôt que u
            qm = seg_vm.sum() / par_m     # probabilité que ce soit v qui meurt plutôt que u
            mem_ed = np.random.binomial(1, qd, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment diffuser
            mem_en = np.random.binomial(1, qn, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment naitre
            mem_em = np.random.binomial(1, qm, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment mourir
            
            mem_ud = np.random.choice(range(len(seg_ud)), p=seg_ud/seg_ud.sum(), size=len(mem_ed)-mem_ed.sum())       
            mem_vd = np.random.choice(range(len(seg_vd)), p=seg_vd/seg_vd.sum(), size=mem_ed.sum()) 
            
            mem_un = np.random.choice(range(len(seg_un)), p=seg_un/seg_un.sum(), size=len(mem_en)-mem_en.sum())       
            mem_vn = np.random.choice(range(len(seg_vn)), p=seg_vn/seg_vn.sum(), size=mem_en.sum()) 
        
            mem_um = np.random.choice(range(len(seg_um)), p=seg_um/seg_um.sum(), size=len(mem_em)-mem_em.sum())       
            mem_vm = np.random.choice(range(len(seg_vm)), p=seg_vm/seg_vm.sum(), size=mem_em.sum()) 

        
    ### Actualisation
        ## Diffusion 
        e, mem_e = mem_e[-1], mem_e[: -1] # e détermine l'espèce qui va diffuser , et mem_e est actualisé sans cette valeur 
        if e == 0: # donc c'est u qui va sauter
            p, mem_u = mem_u[-1], mem_u[: -1] # p détermine le site de u qui va sauter, mem_u est actualisé sans sa dernière valeur qui a été utilisée pour le saut 
        else: # donc c'est v qui va sauter
            p, mem_v = mem_v[-1], mem_v[: -1]  # p détermine quel site de v va sauter, mem_v est actualisé sans sa dernière valeur qui a été utilisée pour le saut 

          
    ### Plus qu'à actualiser les positions avec le saut determiné
        dir = bin[i]  # choisit la direction gauche ou droite (+1 ou -1 pour le moment)
        k = int((p+plage[i]*(1-2*dir))%M) #site d'arrivée
        
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
