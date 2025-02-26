## Partie animation
import matplotlib.animation as animation
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import circular_mean
from sim import simul
from math import ceil

n=0

def Animate(evol,absc,N,Nprime,tauleaping,M,delta,scenario,T,suivi,moy,filename='',length: float = 7): 
    plt.style.use("seaborn-talk")
    xmin = 0
    xmax = 1

    U, V ,D,R= scenario()
    ymax=max(max(U),max(V))
    Uprime = np.zeros((200,M))
    Vprime = np.zeros((200,M))
    Tprime = np.zeros(200)
    Utemp, Vtemp = np.copy(U), np.copy(V)
    iteration = 0

    Ptemp = M//2
    Pprime = np.zeros(200)
    Pprime[0] = M//2
    # On reconstruit notre évolution
    print("Lancement de l'animation")
    for i in tqdm(range(N*tauleaping)):
        
        # Construction de nos populations
        e, k, p, jump_type = evol[i][0],evol[i][1],evol[i][2],evol[i][3]
        k = int(k)
        p = int(p)
        
        if jump_type == 0:
            if e==0:
                Utemp[k] += 1/Nprime
                if Utemp[k]>ymax :
                    ymax = Utemp[k]
                Utemp[p] += -(1/Nprime)
            elif e == 1:
                Vtemp[k] += (1/Nprime)
                if Vtemp[k]>ymax :
                    ymax = Vtemp[k]
                Vtemp[p] += -(1/Nprime)
        if jump_type == 1:
            if e==0:
                Utemp[p] += (1/Nprime)
                if Utemp[p]>ymax :
                    ymax = Utemp[p]
            elif e == 1:
                Vtemp[p] += (1/Nprime)
                if Vtemp[p]>ymax :
                    ymax = Vtemp[p]
        if jump_type == 2:
           if e==0:
            if Utemp[p]>1/Nprime:
                Utemp[p] += -(1/Nprime)
           elif e == 1:
            if Vtemp[p]>1/Nprime:
                Vtemp[p] += -(1/Nprime)
          
        """
        # Suivi position
    
        if suivi == evol[i][0]:
            if evol[i][2] == Ptemp:
                parc = [Utemp, Vtemp]
                if parc[suivi][Ptemp] == 0:
                    pass
                elif test[i]>delta/parc[suivi][Ptemp]:
                    Ptemp = evol[i][1]
        """
        
        if suivi :
            if k == Ptemp:
                random_pop = np.random.binomial(1,1/Nprime*Utemp[p])
                if random_pop == 1 :
                    Ptemp=p

        # Pocessus d'acceleration
        if i % (N*tauleaping//200) == 0:
            Uprime[iteration] = np.copy(Utemp)
            Vprime[iteration] = np.copy(Vtemp)
            Pprime[iteration] = Ptemp
            Tprime[iteration] = T[i]
            iteration +=1
            
    fig , ax = plt.subplots()
    ax.set_xlabel("Space")
    #ax.set_ylabel("Concentrations")
    #Uline, = plt.plot([], [], color ='red') 
    #Vline, = plt.plot([], [], color ='blue')
    Pline, = plt.plot([], [], color ='green', marker='o', markersize=5) 

    plt.xlim(xmin, xmax)
    plt.ylim(-0.1, ymax)
    

    time_text = ax.text(
            0.5,
            0.95,
            "",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    tmax = np.max(Tprime)
    def anim(i):
        # Cas moyenné
        if moy:
            #Uline.set_data(x, moving_average(Uprime[i],M//15))
            #Vline.set_data(x, moving_average(Vprime[i],M//15))
            Uarea = ax.fill_between(absc, np.array(circular_mean(Uprime[i],M//15)), color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, np.array(circular_mean(Vprime[i],M//15)), color="#3f51b5", alpha=0.5)    
            if suivi==1:
                Pline.set_data([absc[int(Pprime[i])]], [circular_mean(Uprime[i],M//15)[int(Pprime[i])]])
    

        # Avec bruit
        else:
            #Uline.set_data(x, Uprime[i])
            #Vline.set_data(x, Vprime[i])
            Uarea = ax.fill_between(absc, Uprime[i], color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, Vprime[i], color="#3f51b5", alpha=0.5)
            if suivi==1:
                Pline.set_data([absc[int(Pprime[i])]], [Uprime[i][int(Pprime[i])]])
        
        
        time_text.set_text("Simulation at t={}s ({}%)".format(
                str(np.round(Tprime[i]*delta, decimals=4)),
                str(int(100 * Tprime[i] / tmax)),
            ))
        
        return Uarea, Varea, Pline, time_text
 
    ani = animation.FuncAnimation(fig, anim, frames= len(Uprime),interval=100, blit=True, repeat=True)
    plt.show()
    
    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------





