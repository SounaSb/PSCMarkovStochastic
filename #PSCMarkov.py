## PSCMarkov

import numpy as np
import matplotlib.pyplot as plt
import math as mt
import copy
from tqdm import tqdm


## Le code qui calcul
def simul(scen):
    U0, V0 = scen()
    T = []
    U = U0.copy()
    V = V0.copy()
    t=0
    
    exp = np.random.exponential(1,N)
    bin = np.random.binomial(1, 0.5, N) # évolution à droite ou à gauche
    plage = np.random.randint(1, 1+ M//40, N)
    
    # Optimisation
    evol = [(2,0,0,3)]
    mem_ud = []
    mem_vd = []
    mem_ed = []
    mem_un = []
    mem_vn = []
    mem_en = []
    mem_um = []
    mem_vm = []
    mem_em = []
    mem_t = []
    
    print("Lancement de la simulation")

    for i in tqdm(range(1,N)):

        if len(mem_t) == 0: # on réactualise si une des listes mémoires est vide
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
            t+=dT
            T.append(t)  # temps du saut i
    
            
            ### type d'évolution
            mem_t=np.random.choice(np.array([0,1,2]), 
                                   p=np.array([par_d/param,par_n/param,par_m/param]),size=2000)    

            ### Qui saute 
            qd = seg_vd.sum() / par_d     # probabilité que ce soit v qui diffuse plutôt que u
            qn = seg_vn.sum() / par_n     # probabilité que ce soit v qui naisse plutôt que u
            qm = seg_vm.sum() / par_m     # probabilité que ce soit v qui meurt plutôt que u
            mem_ed = np.random.binomial(1, qd, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment diffuser
            mem_en = np.random.binomial(1, qn, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment naitre
            mem_em = np.random.binomial(1, qm, size=2000)           # bernoulli de paramètre qd pour choisir qui va effectuverment mourir
            
            mem_ud = np.random.choice(np.arange(len(seg_ud)), p=seg_ud/seg_ud.sum(), size=len(mem_ed)-mem_ed.sum())       
            mem_vd = np.random.choice(np.arange(len(seg_vd)), p=seg_vd/seg_vd.sum(), size=mem_ed.sum()) 
            
            mem_un = np.random.choice(np.arange(len(seg_un)), p=seg_un/seg_un.sum(), size=len(mem_en)-mem_en.sum())       
            mem_vn = np.random.choice(np.arange(len(seg_vn)), p=seg_vn/seg_vn.sum(), size=mem_en.sum()) 
        
            mem_um = np.random.choice(np.arange(len(seg_um)), p=seg_um/seg_um.sum(), size=len(mem_em)-mem_em.sum())       
            mem_vm = np.random.choice(np.arange(len(seg_vm)), p=seg_vm/seg_vm.sum(), size=mem_em.sum()) 

        
        ### Actualisation

        if mem_t[-1] == 0:
            jumptype=0
            ## Determination du site de départ de diffusion et maj
            ed, mem_ed = mem_ed[-1], mem_ed[: -1] # e détermine l'espèce qui va diffuser , et mem_e est actualisé sans cette valeur 
            if ed == 0: # donc c'est u qui va sauter
                pd, mem_ud = mem_ud[-1], mem_ud[: -1] # p détermine le site de u qui va sauter, mem_u est actualisé sans sa dernière valeur qui a été utilisée pour le saut 
            else: # donc c'est v qui va sauter
                pd, mem_vd = mem_vd[-1], mem_vd[: -1]  # p détermine quel site de v va sauter, mem_v est actualisé sans sa dernière valeur qui a été utilisée pour le saut 

            
            ### Plus qu'à actualiser les positions avec le saut determiné
            dir = bin[i]  # choisit la direction gauche ou droite (+1 ou -1 pour le moment)
            kd = int((pd+plage[i]*(1-2*dir))%M) #site d'arrivée
            
            if ed == 0:
                if U[pd]> (M*delta) :
                    U[kd] += (M*delta)
                    U[pd] += -(M*delta)
            else:
                if V[pd] > (M*delta):
                    V[kd] += (M*delta)
                    V[pd] += -(M*delta)

            evol.append((ed, kd, pd,jumptype))
            
        elif (mem_t[-1]==1):
            jumptype=1
            ## determination du site de naissance et maj
            en, mem_en = mem_en[-1], mem_en[: -1]
            if en == 0: # donc c'est u qui va croitre
                pn, mem_un = mem_un[-1], mem_un[: -1] # p détermine le site de u qui va croitre, mem_u est actualisé sans sa dernière valeur qui a été utilisée pour la naissance 
            else: # donc c'est v qui va croitre
                pn, mem_vn = mem_vn[-1], mem_vn[: -1]  # p détermine quel site de v va croitre, mem_v est actualisé sans sa dernière valeur qui a été utilisée pour la naissance 
            
            kn=pn
            ## Naissance
            if en == 0:
                U[pn] += (M*delta)
            else:
                V[pn] += (M*delta)
            evol.append((en, kn, pn,jumptype))
        
        elif (mem_t[-1]==2):
            jumptype=2
            ## determination du site de mort et de l'espèce et maj
            em, mem_em = mem_em[-1], mem_em[: -1]
            if em == 0: # donc c'est u qui va mourir
                pm, mem_um = mem_um[-1], mem_um[: -1] # p détermine le site de u qui va mourir, mem_u est actualisé sans sa dernière valeur qui a été utilisée pour la mort 
            else: # donc c'est v qui va mourir
                pm, mem_vm = mem_vm[-1], mem_vm[: -1]  # p détermine quel site de v va mourir, mem_v est actualisé sans sa dernière valeur qui a été utilisée pour la mort 
            
            km=pm
            ## mort
            if em == 0:
                if U[pm]> (M*delta) :
                    U[pm] += -(M*delta)
            else:
                if V[pm]> (M*delta) :
                    V[pm] += -(M*delta)
            evol.append((em, km, pm,jumptype))
        mem_t= mem_t[:-1]

    print(t)
    print(T[-1])
    return evol, T







#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Partie animation
import matplotlib.animation as animation
import time
n=0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), mode='same')/w

def Animate(evol,scenario,T,suivi,moy,filename=''): 
    plt.style.use("seaborn-talk")
    
    xmin = 0
    xmax = 1

    U, V = scenario()
    Uprime = [U]
    Vprime = [V]
    Utemp, Vtemp = U.copy(), V.copy()

    Ptemp = [M//2]
    Pprime = [M//2]
    test = np.random.uniform(0,1, size=N)

    # On reconstruit notre évolution
    print("Lancement de l'animation")
    for i in tqdm(range(N)):
        
        # Construction de nos populations
        e, k, p, jump_type = evol[i]
        if jump_type == 0:
            if e==0:
                Utemp[k] += (M*delta)
                Utemp[p] += -(M*delta)
            elif e == 1:
                Vtemp[k] += (M*delta)
                Vtemp[p] += -(M*delta)
        if jump_type == 1:
            if e==0:
                Utemp[p] += (M*delta)
            elif e == 1:
                Vtemp[p] += (M*delta)
        if jump_type == 2:
           if e==0:
            if Utemp[p]>(M*delta):
                Utemp[p] += -(M*delta)
           elif e == 1:
            if Vtemp[p]>(M*delta):
                Vtemp[p] += -(M*delta)
          

        # Suivi position
        if suivi == evol[i][0]:
            if evol[i][2] == Ptemp:
                parc = [Utemp, Vtemp]
                if parc[suivi][Ptemp] == 0:
                    pass
                elif test[i]>1/parc[suivi][Ptemp]:
                    Ptemp = evol[i][1]

        # Pocessus d'acceleration
        if i % (N//250) == 0:
            Uprime.append(Utemp.copy())
            Vprime.append(Vtemp.copy())
            Pprime.append(Ptemp)

    fig , ax = plt.subplots()
    ax.set_xlabel("Space")
    ax.set_ylabel("Concentrations")
    #Uline, = plt.plot([], [], color ='red') 
    #Vline, = plt.plot([], [], color ='blue')
    Pline, = plt.plot([], [], color ='green', marker='o', markersize=5) 

    plt.xlim(xmin, xmax)
    plt.ylim(-0.1, 8)
    

    st = plt.suptitle("", fontweight="bold")

    def anim(i):
        # Cas moyenné
        if moy:
            #Uline.set_data(x, moving_average(Uprime[i],M//15))
            #Vline.set_data(x, moving_average(Vprime[i],M//15))
            Uarea = ax.fill_between(absc, moving_average(Uprime[i],M//15), color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, moving_average(Vprime[i],M//15), color="#3f51b5", alpha=0.5)    
            if suivi==0:
                Pline.set_data(absc[Pprime[i]], [moving_average(Uprime[i],M//17)[Pprime[i]]])
            else:
                Pline.set_data(absc[Pprime[i]], [moving_average(Vprime[i],M//17)[Pprime[i]]])

        # Avec bruit
        else:
            #Uline.set_data(x, Uprime[i])
            #Vline.set_data(x, Vprime[i])
            Uarea = ax.fill_between(absc, Uprime[i], color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, Vprime[i], color="#3f51b5", alpha=0.5)
            
            if suivi==0:
                Pline.set_data(absc[Pprime[i]], [Uprime[i][Pprime[i]]])
            else:
                Pline.set_data(absc[Pprime[i]], [Vprime[i][Pprime[i]]])
        
        
        st.set_text("Population dynamics simulation at t={}s".format(
                    str(np.round(T[i], decimals=2))
                ))
        
        print(T[i])
        return Uarea, Varea, Pline
 
    ani = animation.FuncAnimation(fig, anim, frames= len(Uprime),interval=1, blit=True, repeat=True)
    plt.show()
    
    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------





## On exécute maintenant
D=[[0.005, 0, 3], [0.005, 0, 0]]
R=[[5, 3, 1], [2, 1, 3]]
M = 200
N = 2000000
delta = 1e-5
absc = np.linspace(start = 0, stop = 1, num = M)


## Scenarios
# Intru scenario
def intru():
    V0 = 0*absc+np.rint(0.8/(M*delta))*(M*delta)
    U0 = 0*absc+np.rint(0.2/(M*delta))*(M*delta)
    for i in range(int(0.2*M)):
        V0[i+int(0.4*M)] += np.rint(1.5/(M*delta))*(M*delta)
    return U0, V0

## sin dec:
def sindec():
    U0 = M*delta*np.rint((1+np.sin(2*np.pi*absc))/(M*delta))
    V0 = M*delta*np.rint((1+np.cos(2*np.pi*absc))/(M*delta))
    return U0, V0

def test():
    U0=1 + np.cos(2 * np.pi * absc)
    V0=1 + np.sin(2 * np.pi * absc)
    return U0,V0
    
# Type de simulation
simul_type = test
# On calcul
evol, T = simul(simul_type)
# On anime
Animate(evol, simul_type, T, suivi=0, moy=False)

