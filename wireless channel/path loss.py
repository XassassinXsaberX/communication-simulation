import matplotlib.pyplot as plt
import numpy as np


def PL_free_space(fc, dist, Gt=1, Gr=1):
    l = 3*(10**8)/fc #波長
    PL = -10*np.log10(Gt*Gr*l*l/(4*np.pi*4*np.pi*dist*dist))
    return PL

def PL_log(fc,dist,d0,n,sigma=0):
    l = 3*(10**8)/fc #波長
    PL = PL_free_space(fc,d0) + 10*n*np.log10(dist/d0) + sigma*np.random.randn()
    return PL

d = [0]*11
PL = [0]*11
fc = 1.5*(10**9)
d0 = 100
sigma = 3
d[0] = 1
i=1
while True:
    if i==len(d) :
       break
    d[i] = d[i-1]*2
    i+=1

plt.figure('Free-space path loss model')
for i in range(len(d)):
    PL[i] = PL_free_space(fc,d[i])
plt.semilogx(d,PL,marker='<',label='Gt=1,Gr=1')
for i in range(len(d)):
    PL[i] = PL_free_space(fc,d[i],1,0.5)
plt.semilogx(d,PL,marker='^',label='Gt=1,Gr=0.5')
for i in range(len(d)):
    PL[i] = PL_free_space(fc,d[i],0.5,0.5)
plt.semilogx(d,PL,marker='o',label='Gt=0.5,Gr=0.5')
plt.legend()
plt.grid(True,which='both')
plt.xlabel('distance (m)')
plt.ylabel('path loss (dB)')
plt.title('Free-space path loss model   fc=1.5GHz')

plt.figure('log-distance path loss model')
for j in range(3):
    if j==0:
        n=2
    elif j==1:
        n=3
    else:
        n=6
    for i in range(len(d)):
        PL[i] = PL_log(fc,d[i],d0,n)
    plt.semilogx(d,PL,marker='o',label='n={0}'.format(n))
plt.legend()
plt.ylim(40)
plt.grid(True,which='both')
plt.xlabel('distance (m)')
plt.ylabel('path loss (dB)')
plt.title('log-distance path loss model   fc=1.5GHz')

plt.figure('log-normal shadowing path loss model')
n = 2
for j in range(3):
    for i in range(len(d)):
        PL[i] = PL_log(fc,d[i],d0,n,sigma)
    plt.semilogx(d,PL,marker='o',label='path {0}'.format(j+1))
plt.legend()
plt.ylim(40)
plt.grid(True,which='both')
plt.xlabel('distance (m)')
plt.ylabel('path loss (dB)')
plt.title('log-normal shadowing path loss model   fc=1.5GHz, std=3dB, n=2')
plt.show()

