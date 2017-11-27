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
#visited_node5 = [200.956206, 193.302858, 188.23704 ,184.28118 ,181.40763 ,179.280066, 177.685578, 177.311892, 176.823498, 176.579904 ,176.45169, 176.883222, 176.49669  ]
#plt.plot(snr,visited_node5,marker = 'o',label='soft = 6')
#visited_node5 = [279.894993, 269.276574 ,262.765482, 257.993414, 254.564884 ,251.646374, 250.090099 ,249.306288 ,248.879904 ,248.82109, 248.739547, 248.652047, 249.008088 ]
#plt.plot(snr,visited_node5,marker = 'o',label='soft = 7')
#visited_node5 = [370.293768 ,356.888248, 349.157272, 343.346768, 338.926976 ,335.839872 ,333.511592 ,332.104064, 331.771808, 331.730224, 331.130096 ,331.853416, 332.472296 ]
#plt.plot(snr,visited_node5,marker = 'o',label='soft = 8')


plt.xlabel('Eb/No , dB')
plt.ylabel('Multiplication complexity')
plt.grid(True, which='both')
plt.legend()
plt.show()