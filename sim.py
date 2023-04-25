import numpy as np
from tqdm import tqdm
from scipy.integrate import quad


## Le code qui calcul
def simul(scen,N,Nprime,tauleaping,M,delta):
    U0, V0 ,D,R = scen()
    x=np.arange(M)
    
    R[0][0]/=float(M)
    R[0][1]/=float(M)
    R[0][2]/=float(M)
    R[1][0]/=float(M)
    R[1][1]/=float(M)
    R[1][2]/=float(M)
        
    U0 = np.floor(Nprime*U0)/Nprime
    V0 = np.floor(Nprime*V0)/Nprime
    T = np.zeros(N*tauleaping)
    U = U0.copy()
    V = V0.copy()
    t=0

    exp = np.random.exponential(1,N)
    bin = np.random.binomial(1, 0.5, N*tauleaping) # évolution à droite ou à gauche
    plage = np.random.randint(1, 1+ M//40, N)
    gauss = np.random.normal(0,M//40,N)
    
    # Optimisation
    evol = np.zeros((N*tauleaping,4))
    evol[0] = np.array([2,0,0,3])
    
    print("Lancement de la simulation")

    for i in tqdm(range(N)):
    
        
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
        
        dT = exp[i] / param # intervalle de temps jusqu'au prochain saut
        t+=dT
        # temps du saut i
        
        ### type d'évolution
        mem_t=np.random.choice(np.array([0,1,2]), 
                                p=np.array([par_d/param,par_n/param,par_m/param]),
                                size=tauleaping)

        ### Qui saute 
        if not(par_d>0):
            qd=0
        else :
            qd = seg_vd.sum() / par_d     # probabilité que ce soit v qui diffuse plutôt que u
        if not(par_n>0):
            qn=0
        else :
            qn=seg_vn.sum() / par_n     # probabilité que ce soit v qui naisse plutôt que u
        if not(par_m>0):
            qm=0
        else :
            qm=seg_vm.sum() / par_m     # probabilité que ce soit v qui meurt plutôt que u
        
        if not (qd>0):
            mem_ed = np.zeros(tauleaping)
        else :
            mem_ed = np.random.binomial(1, qd, size=tauleaping)           # bernoulli de paramètre qd pour choisir qui va effectuverment diffuser
        if not (qn>0):
            mem_en = np.zeros(tauleaping)
        else :
            mem_en = np.random.binomial(1, qn, size=tauleaping)           # bernoulli de paramètre qd pour choisir qui va effectuverment naitre
        if not (qm>0):
            mem_em = np.zeros(tauleaping)
        else :
            mem_em = np.random.binomial(1, qm, size=tauleaping)           # bernoulli de paramètre qd pour choisir qui va effectuverment mourir
    
        ### Position de départ de diffusion
        if not (np.sum(seg_ud)>0):
            mem_ud = np.random.randint(M,tauleaping)
        else :
            mem_ud = np.random.choice(x, p=seg_ud/np.sum(seg_ud), size=tauleaping)
            
        if not (np.sum(seg_vd)>0):
            mem_vd = np.random.randint(M,tauleaping)
        else :
            mem_vd = np.random.choice(x, p=seg_vd/np.sum(seg_vd), size=tauleaping)
        
        if not (np.sum(seg_un)>0):
            mem_un = np.random.randint(M,tauleaping)
        else :
            mem_un = np.random.choice(x, p=seg_un/np.sum(seg_un), size=tauleaping)
        
        if not (np.sum(seg_vn)>0):
            mem_vn = np.random.randint(M,tauleaping)
        else :
            mem_vn = np.random.choice(x, p=seg_vn/np.sum(seg_vn), size=tauleaping)
        
        if not (np.sum(seg_um)>0):
            mem_um = np.random.randint(M,tauleaping)
        else :
            mem_um = np.random.choice(x, p=seg_um/np.sum(seg_um), size=tauleaping)
        
        if not (np.sum(seg_vm)>0):
            mem_vm = np.random.randint(M,tauleaping)
        else :
            mem_vm = np.random.choice(x, p=seg_vm/np.sum(seg_vm), size=tauleaping)
        
    
        
        ### Actualisation
        for j in range (tauleaping):
            if mem_t[j] == 0:
                jumptype=0
                ## Determination du site de départ de diffusion et maj
                ed= mem_ed[j]# e détermine l'espèce qui va diffuser 
                if ed == 0: # donc c'est u qui va sauter
                    pd = mem_ud[j] # p détermine le site de u qui va sauter, mem_u est actualisé sans sa dernière valeur qui a été utilisée pour le saut 
                else: # donc c'est v qui va sauter
                    pd = mem_vd[j]  # p détermine quel site de v va sauter, mem_v est actualisé sans sa dernière valeur qui a été utilisée pour le saut 

                
                ### Plus qu'à actualiser les positions avec le saut determiné
                dir = bin[i]  # choisit la direction gauche ou droite (+1 ou -1 pour le moment)
                kd = int((pd+plage[i]*(1-2*dir))%M) #site d'arrivée
                kd = int((pd+gauss[i])%M)
                
                if ed == 0:
                    if U[pd]> (1/Nprime) :
                        U[kd] += (1/Nprime)
                        U[pd] += -(1/Nprime)
                else:
                    if V[pd] > (1/Nprime):
                        V[kd] += (1/Nprime)
                        V[pd] += -(1/Nprime)

                evol[int(i*tauleaping+j)]=np.array([ed, kd, pd,jumptype])
                
            elif (mem_t[j]==1):
                jumptype=1
                ## determination du site de naissance et maj
                en = mem_en[j]
                if en == 0: # donc c'est u qui va croitre
                    pn = mem_un[j] # p détermine le site de u qui va croitre
                else: # donc c'est v qui va croitre
                    pn = mem_vn[j]  # p détermine quel site de v va croitre
                
                kn=pn
                ## Naissance
                if en == 0:
                    U[pn] += (1/Nprime)
                else:
                    V[pn] += (1/Nprime)
                evol[int(i*tauleaping + j)]=np.array([en, kn, pn,jumptype])
            
            elif (mem_t[j]==2):
                jumptype=2
                ## determination du site de mort et de l'espèce et maj
                em = mem_em[j]
                if em == 0: # donc c'est u qui va mourir
                    pm = mem_um[j]  # p détermine le site de u qui va mourir
                else: # donc c'est v qui va mourir
                    pm = mem_vm[j]  # p détermine quel site de v va mourir
                
                km=pm
                ## mort
                if em == 0:
                    if U[pm]> (1/Nprime) :
                        U[pm] += -(1/Nprime)
                else:
                    if V[pm]> (1/Nprime) :
                        V[pm] += -(1/Nprime)
                evol[int(i*tauleaping + j)]=np.array([em, km, pm,jumptype])
            T[int(i*tauleaping + j)]=t
    return evol, T





