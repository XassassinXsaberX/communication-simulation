import numpy as np
import matplotlib.pyplot as plt

#QPSK
snr = [0 ,2 ,4 ,6 ,8 ,10, 12 ,14 ,16, 18, 20, 22, 24 ]
#16QAM
#snr = [0.0 ,2.5 ,5.0 ,7.5, 10.0 ,12.5 ,15.0, 17.5, 20.0, 22.5 ,25.0 ,27.5 ,30.0  ]
#64QAM
#snr = [0 ,3 ,6 ,9, 12, 15 ,18, 21, 24, 27, 30 ,33, 36]

ber1 = [13 ]*13
plt.plot(snr,ber1,marker = 'o',label='branch vector = [1,1,1,1] ,K=2')
ber2 = [26 ]*13
plt.plot(snr,ber2,marker = 'o',label='branch vector = [1,1,1,2] ,K=2')
ber3 = [32 ]*13
plt.plot(snr,ber3,marker = 'o',label='branch vector = [1,1,2,2] ,K=2')
ber4 = [40]*13
plt.plot(snr,ber4,marker = 'o',label='branch vector = [1,2,2,2] ,K=2')
ber5 = [50 ]*13
plt.plot(snr,ber5,marker = 'o',label='branch vector = [2,2,2,2] ,K=2')
ber5 = [86 ]*13
plt.plot(snr,ber5,marker = 'o',label='branch vector = [2,2,2,2] ,K=4')
#add_computation4 = [1349.787698 ,1295.704958 ,1260.489616 ,1233.277024 ,1214.25209, 1200.429686 ,1190.065134 ,1187.825498, 1184.940992 ,1183.37747 ,1182.735032, 1185.843752, 1183.205036   ]
#plt.plot(snr,add_computation4,marker = 'o',label='soft = 6')
#add_computation4 = [1930.310519, 1853.64131, 1807.496266, 1774.182547, 1750.945718, 1731.441064 ,1721.458885 ,1716.679343 ,1713.988161, 1713.806529, 1713.398746, 1713.05227, 1715.52793 ]
#plt.plot(snr,add_computation4,marker = 'o',label='soft = 7')
#add_computation4 = [2604.452262 ,2506.385546, 2450.637898, 2409.517836 ,2379.496456, 2358.301144 ,2342.679694, 2333.378864, 2331.83376, 2331.566292, 2327.578292 ,2332.791846, 2337.492606 ]
#plt.plot(snr,add_computation4,marker = 'o',label='soft = 8')


plt.xlabel('Eb/No , dB')
plt.ylabel('Addition complexity')
plt.grid(True, which='both')
plt.legend()
plt.show()