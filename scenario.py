import numpy as np
from sim import simul
from anim import Animate
from utils import plateau


## On exécute maintenant

M = 200
Nprime = 1000 # grandeur caractéristique de la population

N = 5000
tauleaping = 5000 #compléxité en N*tauleaping

delta = 1e-5
absc = np.linspace(start = 0, stop = 1, num = M)


## Scenarios
# Intru scenario
def intru():
    D=np.array([[0.005, 0, 3], [0.005, 0, 0]])
    R=np.array([[5, 3, 1], [2, 1, 3]])
    V0 = 0*absc+np.rint(0.8/(M*delta))*(M*delta)
    U0 = 0*absc+np.rint(0.2/(M*delta))*(M*delta)
    for i in range(int(0.2*M)):
        V0[i+int(0.4*M)] += np.rint(1.5/(M*delta))*(M*delta)
    return U0, V0 , D, R

## sin dec:
def sindec():
    D=np.array([[0.005, 0, 3],
                [0.005, 0, 0]])
    R=np.array([[5, 3, 1], 
                [2, 1, 3]])
    U0 = M*delta*np.rint((1+np.sin(2*np.pi*absc))/(M*delta))
    V0 = M*delta*np.rint((1+np.cos(2*np.pi*absc))/(M*delta))
    return U0, V0, D, R

def test():
    D=np.array([[0.005, 0, 3], 
                [0.005, 0, 0]])
    R=np.array([[5, 3, 1], 
                [2, 1, 3]])
    U0=1 + np.cos(2 * np.pi * absc)
    V0=1 + np.sin(2 * np.pi * absc)
    return U0,V0, D, R
def test2():
    D=np.array([[10, 1e-8, 1e-8], 
                [1e-8, 1e-8, 1e-8]])
    R=np.array([[1e-8, 1e-8, 1e-8], 
                [1e-8, 1e-8, 1e-8]])
    U0=1 + np.cos(2 * np.pi * absc)
    V0=1 + np.sin(2 * np.pi * absc)
    return U0,V0, D, R

def test3():
    D=np.array([[10, 10, 10], 
                [5, 5, 5]])
    R=np.array([[1e-8, 1e-8, 1e-8], 
                [1e-8, 1e-8, 1e-8]])
    U0=1 + np.cos(2 * np.pi * absc)
    V0=1 + np.sin(2 * np.pi * absc)
    return U0,V0, D, R

def LinearScenario():
    D=np.array([[0.005, 0, 3], [0.005, 0, 0]])
    R=np.array([[5, 3, 1], [2, 1, 3]])
    U0= .2 + plateau(absc + .1)
    V0= .4 - plateau(absc - .1)
    return U0,V0, D, R
    
def SineScenario1():
    D=np.array([[0.005, 0, 3], [0.005, 0, 0]])
    R=np.array([[5, 3, 1], [2, 1, 3]])
    U0 = np.sin(4 * absc + .12)**2
    V0 = np.sin(4 * absc - .12)**2
    return U0,V0, D, R

def SineScenario2():
    D=np.array([[0.000001, 0, 2e-2], [2e-2, 0.000001, 0]])
    R=np.array([[1e-8, 1e-8, 1e-8], [1e-8, 1e-8, 1e-8]])
    U0=np.cos(2 * absc)**2
    V0=np.sin(6 * absc)**2
    return U0,V0, D, R

def IntruderScenario():
    D=np.array([[0, 0, 2e-2], [0.2e-2, 0, 0]])
    R=np.array([[1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6]])
    U0 = 3*plateau(5*absc)
    V0 =plateau(absc/2)
    return U0,V0, D, R



    
# Type de simulation
simul_type = test
# On calcul
evol, T = simul(simul_type,N,Nprime,tauleaping,M,delta)

# On anime
Animate(evol,absc,N,Nprime,tauleaping,M,delta,simul_type,T, suivi=0, moy=False)