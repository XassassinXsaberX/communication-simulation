import numpy as np
import matplotlib.pyplot as plt

#QPSK
snr = [0 ,2 ,4 ,6 ,8 ,10, 12 ,14 ,16, 18, 20, 22, 24 ]
#16QAM
#snr = [0.0 ,2.5 ,5.0 ,7.5, 10.0 ,12.5 ,15.0, 17.5, 20.0, 22.5 ,25.0 ,27.5 ,30.0  ]
#64QAM
#snr = [0 ,3 ,6 ,9, 12, 15 ,18, 21, 24, 27, 30 ,33, 36]

ber1 = [4 ]*13
plt.plot(snr,ber1,marker = 'o',label='branch vector = [1,1,1,1] ,K=2')
ber2 = [8 ]*13
plt.plot(snr,ber2,marker = 'o',label='branch vector = [1,1,1,2] ,K=2')
ber3 = [10 ]*13
plt.plot(snr,ber3,marker = 'o',label='branch vector = [1,1,2,2] ,K=2')
ber4 = [12]*13
plt.plot(snr,ber4,marker = 'o',label='branch vector = [1,2,2,2] ,K=2')
ber5 = [14 ]*13
plt.plot(snr,ber5,marker = 'o',label='branch vector = [2,2,2,2] ,K=2')
ber5 = [22 ]*13
plt.plot(snr,ber5,marker = 'o',label='branch vector = [2,2,2,2] ,K=4')



plt.xlabel('Eb/No , dB')
plt.ylabel('Average visited node')
plt.grid(True, which='both')
plt.legend()
plt.show()