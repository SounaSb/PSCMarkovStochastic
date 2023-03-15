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
    evol = [(2,0,0,3)]
    mem_u = []
    mem_v = []
    mem_e = []

    dT = 0  

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
            T[i] = T[i-1]+dT # temps du saut i
            
            
            ### type d'évolution
            
            
            mem_t=np.random.choice([0,1,2],
                                [par_d/param,par_n/param,par_m/param],2000)    

            ### Qui saute
            
            
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

        if mem_t[i] == 0:
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
                if U[pd]> 0 :
                    U[kd] += 1
                    U[pd] += -1
            else:
                if V[pd] > 0:
                    V[kd] += 1
                    V[pd] += -1

            evol.append((ed, kd, pd,jumptype))
            
        elif (mem_t[i]==1):
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
                U[pn] += 1
            else:
                V[pn] += 1
            evol.append((en, kn, pn,jumptype))
        
        elif (mem_t[i]==2):
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
                U[pn] += 1
            else:
                V[pn] += 1
            evol.append((en, kn, pn,jumptype))
  
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
