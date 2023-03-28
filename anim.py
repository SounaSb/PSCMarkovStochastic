## Partie animation
import matplotlib.animation as animation
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import circular_mean
from sim import simul
n=0


def Animate(evol,absc,N,M,delta,scenario,T,suivi,moy,filename='',length: float = 7): 
    plt.style.use("seaborn-talk")
    
    xmin = 0
    xmax = 1

    U, V = scenario()
    Uprime = [U]
    Vprime = [V]
    Tprime = [T[0]]
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
                elif test[i]>delta/parc[suivi][Ptemp]:
                    Ptemp = evol[i][1]

        # Pocessus d'acceleration
        if i % (N//250) == 0:
            Uprime.append(Utemp.copy())
            Vprime.append(Vtemp.copy())
            Pprime.append(Ptemp)
            Tprime.append(T[i])

    fig , ax = plt.subplots()
    ax.set_xlabel("Space")
    ax.set_ylabel("Concentrations")
    #Uline, = plt.plot([], [], color ='red') 
    #Vline, = plt.plot([], [], color ='blue')
    Pline, = plt.plot([], [], color ='green', marker='o', markersize=5) 

    plt.xlim(xmin, xmax)
    plt.ylim(-0.1, 8)
    

    time_text = ax.text(
            0.5,
            0.95,
            "",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    def anim(i):
        
        # Cas moyenné
        if moy:
            #Uline.set_data(x, moving_average(Uprime[i],M//15))
            #Vline.set_data(x, moving_average(Vprime[i],M//15))
            Uarea = ax.fill_between(absc, circular_mean(Uprime[i],M//15), color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, circular_mean(Vprime[i],M//15), color="#3f51b5", alpha=0.5)    
            if suivi==0:
                Pline.set_data(absc[Pprime[i]], [circular_mean(Uprime[i],M//17)[Pprime[i]]])
            else:
                Pline.set_data(absc[Pprime[i]], [circular_mean(Vprime[i],M//17)[Pprime[i]]])

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
        
        
        time_text.set_text("Population dynamics simulation at t={}s".format(
                    str(np.round(Tprime[i], decimals=2))
                ))
        return Uarea, Varea, Pline, time_text
 
    ani = animation.FuncAnimation(fig, anim, frames= len(Uprime),interval=1, blit=True, repeat=True)
    plt.show()
    
    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)


def Animate(evol,absc,N,M,delta,scenario,T,suivi,moy,filename='',length: float = 7): 
    plt.style.use("seaborn-talk")
    
    xmin = 0
    xmax = 1

    U, V ,D,R= scenario()
    ymax=max(max(U),max(V))
    Uprime = [U]
    Vprime = [V]
    Tprime = [T[0]]
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
                if Utemp[k]>ymax :
                    ymax = Utemp[k]
                Utemp[p] += -(M*delta)
            elif e == 1:
                Vtemp[k] += (M*delta)
                if Vtemp[k]>ymax :
                    ymax = Vtemp[k]
                Vtemp[p] += -(M*delta)
        if jump_type == 1:
            if e==0:
                Utemp[p] += (M*delta)
                if Utemp[p]>ymax :
                    ymax = Utemp[p]
            elif e == 1:
                Vtemp[p] += (M*delta)
                if Vtemp[p]>ymax :
                    ymax = Vtemp[p]
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
                elif test[i]>delta/parc[suivi][Ptemp]:
                    Ptemp = evol[i][1]

        # Pocessus d'acceleration
        if i % (N//250) == 0:
            Uprime.append(Utemp.copy())
            Vprime.append(Vtemp.copy())
            Pprime.append(Ptemp)
            Tprime.append(T[i])

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

    def anim(i):
        
        # Cas moyenné
        if moy:
            #Uline.set_data(x, moving_average(Uprime[i],M//15))
            #Vline.set_data(x, moving_average(Vprime[i],M//15))
            Uarea = ax.fill_between(absc, circular_mean(Uprime[i],M//15), color="#f44336", alpha=0.5)
            Varea = ax.fill_between(absc, circular_mean(Vprime[i],M//15), color="#3f51b5", alpha=0.5)    
            if suivi==0:
                Pline.set_data(absc[Pprime[i]], [circular_mean(Uprime[i],M//17)[Pprime[i]]])
            else:
                Pline.set_data(absc[Pprime[i]], [circular_mean(Vprime[i],M//17)[Pprime[i]]])

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
        
        
        time_text.set_text("Population dynamics simulation at t={}s".format(
                    str(np.round(Tprime[i]*delta, decimals=2))
                ))
        return Uarea, Varea, Pline, time_text
 
    ani = animation.FuncAnimation(fig, anim, frames= len(Uprime),interval=1, blit=True, repeat=True)
    plt.show()
    
    if filename:
        w = animation.writers['ffmpeg']
        w = animation.FFMpegWriter(fps=60, bitrate=1800)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------





