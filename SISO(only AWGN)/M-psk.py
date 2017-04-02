import numpy as np
import matplotlib.pyplot as plt
import math

snr_db = [0]*13
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(5):#總共有BPSK  QPSK  8-PSK三種調變
    for i in range(len(snr)):
        if k==1:#bpsk theory
            ber[i] = 1/2*math.erfc(np.sqrt(snr[i]))
            continue
        elif k==4:#8-psk theory
            ber[i] = 1/3*math.erfc(np.sqrt(3*snr[i])*np.sin(np.pi/8))
            continue
        error = 0
        for j in range(N):
            if k == 0 :#bpsk
                # 決定bpsk的2個星座點
                constellation = [-1,1]
                No = 1/snr[i]
            elif k == 2:  # qpsk
                # 決定qsk的4個星座點
                # SNR = Eb / No
                # 若星座點為1+1j，則Es = 2，Eb = 1，故SNR = 1 / No
                constellation = [1+1j, 1-1j, -1+1j, -1-1j]
                No = 1/snr[i]
            elif k == 3:  # 8-psk
                # 決定8-psk的8個星座點
                # SNR = Eb / No
                # 若Es = 1，因為Es = 3*Eb。所以Eb = 1/3，故SNR = (1/3) / No
                constellation = [0]*8
                No = (1/3) / snr[i]
                for m in range(len(constellation)):
                    constellation[m] = np.cos(2*np.pi/8*(m-1))-1j*np.sin(2*np.pi/8*(m-1))


            b = np.random.random()# 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
            for m in range(len(constellation)):
                if b <= (m+1)/len(constellation):
                    symbol = constellation[m]
                    break


            # 接下來加上雜訊
            receive = symbol + np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

            # 接收端利用Maximum Likelihood來detect symbol
            min_distance = 10 ** 9
            for m in range(len(constellation)):
                if abs(constellation[m] - receive) < min_distance:
                    detection = constellation[m]
                    min_distance = abs(constellation[m] - receive)

            # 紀錄錯幾個symbol
            if detection != symbol:
                error += 1


        if k == 0:   #bpsk的ber
            ber[i] = error / N
        elif k == 2: #qpsk的ber = symbol error rate / 2
            ber[i] = error / (2*N)
        elif k == 3: #8-psk的ber = symbol error rate / 3
            ber[i] = error / (3*N)

    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='bpsk (simulation)')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='bpsk (theory)')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='qpsk (simulation)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='8-psk (simulation)')
    elif k==4:
        plt.semilogy(snr_db,ber,marker='o',label='8-psk (theory)')

plt.grid(True,which='both')
plt.legend()
plt.xlabel('SNR(Eb/No)  (dB)')
plt.ylabel('ber')
plt.show()
