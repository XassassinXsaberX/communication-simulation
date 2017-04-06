import numpy as np
import matplotlib.pyplot as plt
import math

snr_db = [0]*11
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(4):#總共有BPSK  QPSK  8-PSK三種調變
    for i in range(len(snr)):
        if k==1:#BPSK rayleigh theory
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue
        error = 0
        for j in range(N):
            if k == 0 :#BPSK
                # 決定BPSK的2個星座點
                constellation = [-1,1]
                No = 1/snr[i]
            elif k == 2:  # QPSK
                # 決定qsk的4個星座點
                # SNR = Eb / No
                # 若星座點為1+1j，則Es = 2，Eb = 1，故SNR = 1 / No
                constellation = [1+1j, 1-1j, -1+1j, -1-1j]
                No = 1/snr[i]
            elif k == 3:  # 8-PSK
                # 決定8-PSK的8個星座點
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

            # 接下來考慮rayleigh fading
            h = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()
            receive = symbol * h

            # 接下來加上雜訊
            receive = receive + np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

            # 接收端先用matched filter來解調信號，再用Maximum Likelihood來detect symbol
            receive = receive / h
            min_distance = 10 ** 9
            for m in range(len(constellation)):
                if abs(constellation[m] - receive) < min_distance:
                    detection = constellation[m]
                    min_distance = abs(constellation[m] - receive)

            # 紀錄錯幾個symbol
            if detection != symbol:
                if k==2 :#要確實的找出QPSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                    if abs(symbol.real - detection.real) == 2:
                        error += 1
                    if abs(symbol.imag - detection.imag) == 2:
                        error += 1
                elif k==3 :#要確實的找出8-PSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                    # 這裡我們用向量間夾角的餘弦函式cos來判斷錯幾個bit
                    cos_rad = (symbol.real*detection.real + symbol.imag*detection.imag) / (abs(symbol)*abs(detection))
                    if abs( cos_rad - np.cos(np.pi/4) ) < 0.001:
                        error += 1
                    elif abs( cos_rad - np.cos(np.pi/2) ) < 0.001:
                        error += 2
                    elif abs( cos_rad - np.cos(np.pi*3/4) ) < 0.001:
                        error += 3
                    elif abs( cos_rad - np.cos(np.pi) ) < 0.001:
                        error += 2
                else:
                    error += 1



        if k == 0:   #BPSK的ber
            ber[i] = error / N
        elif k == 2: #QPSK的ber
            ber[i] = error / (2*N) #除上(2*N)是因為一個QPSK symbol含有個bit，所以送N個symbol時，實際上是送(2*N)個bit
        elif k == 3: #8-PSK的ber
            ber[i] = error / (3*N) #除上(3*N)是因為一個QPSK symbol含有個bit，所以送N個symbol時，實際上是送(3*N)個bit

    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='BPSK (simulation)')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='BPSK (theory)')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='QPSK (simulation)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='8-PSK (simulation)')

plt.title('BER in Rayleigh fading')
plt.grid(True,which='both')
plt.legend()
plt.xlabel('SNR(Eb/No)  (dB)')
plt.ylabel('ber')
plt.show()
