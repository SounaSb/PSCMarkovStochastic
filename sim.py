import numpy as np
from tqdm import tqdm


## Le code qui calcul
def simul(scen,N,M,delta):
    U0, V0 ,D,R = scen()
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

    for i in tqdm(range(1,N)):
    
        dT = exp[i] / param # intervalle de temps jusqu'au prochain saut
        t+=dT
        T.append(t)  # temps du saut i

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
            
        
    
            
            ### type d'évolution
            mem_t=np.random.choice(np.array([0,1,2]), 
                                   p=np.array([par_d/param,par_n/param,par_m/param]),
                                   size=2000)    

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
    return evol, T





