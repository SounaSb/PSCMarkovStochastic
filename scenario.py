import numpy as np
from sim import simul
from anim import Animate

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
evol, T = simul(simul_type,N,M,D,R,delta)
# On anime
Animate(evol, simul_type, T, suivi=0, moy=False)