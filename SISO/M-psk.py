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
        if k==1:#BPSK theory
            ber[i] = 1/2*math.erfc(np.sqrt(snr[i]))
            continue
        elif k==4:#8-PSK theory
            ber[i] = 1/3*math.erfc(np.sqrt(3*snr[i])*np.sin(np.pi/8))
            continue
        error = 0
        for j in range(N):
            if k == 0 :#BPSK
                # 決定BPSK的2個星座點
                constellation = [-1,1]
            elif k == 2:  # QPSK
                # 決定qsk的4個星座點
                constellation = [1+1j, 1-1j, -1+1j, -1-1j]
            elif k == 3:  # 8-PSK
                # 決定8-PSK的8個星座點
                constellation = [0]*8
                for m in range(len(constellation)):
                    constellation[m] = np.cos(2*np.pi/8*(m-1))-1j*np.sin(2*np.pi/8*(m-1))

            K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
            # 接下來要算平均一個symbol有多少能量
            # 先將所有可能的星座點能量全部加起來
            energy = 0
            for m in range(len(constellation)):
                energy += abs(constellation[m]) ** 2
            Es = energy / len(constellation)      # 平均一個symbol有Es的能量
            Eb = Es / K                           # 平均一個bit有Eb能量
            No = Eb / snr[i]                      # 決定No


            b = np.random.random()# 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
            for m in range(len(constellation)):
                if b <= (m+1)/len(constellation):
                    symbol = constellation[m]
                    break


            # 接下來加上雜訊
            receive = symbol + np.sqrt(No/2) * np.random.randn() + 1j*np.sqrt(No/2) * np.random.randn()

            # 接收端利用Maximum Likelihood來detect symbol
            min_distance = 10 ** 9
            for m in range(len(constellation)):
                if abs(constellation[m] - receive) < min_distance:
                    detection = constellation[m]
                    min_distance = abs(constellation[m] - receive)

            # 紀錄錯幾個symbol
            if detection != symbol:
                if k == 2 :# 要確實的找出QPSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                    if abs(detection.real-symbol.real) == 2 :
                        error += 1
                    if abs(detection.imag-symbol.imag) == 2 :
                        error += 1
                elif k==3 :# 要確實的找出8-PSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                    # 這裡我們用向量間夾角的餘弦函式cos來判斷錯幾個bit
                    cos_rad = (symbol.real * detection.real + symbol.imag * detection.imag) / (abs(symbol) * abs(detection))
                    if abs(cos_rad - np.cos(np.pi / 4)) < 0.001:
                        error += 1
                    elif abs(cos_rad - np.cos(np.pi / 2)) < 0.001:
                        error += 2
                    elif abs(cos_rad - np.cos(np.pi * 3 / 4)) < 0.001:
                        error += 3
                    elif abs(cos_rad - np.cos(np.pi)) < 0.001:
                        error += 2
                else:
                    error += 1


        ber[i] = error / (K*N) #因為平均一個symbol含有K個bit

    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='BPSK (simulation)')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='BPSK (theory)')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='QPSK (simulation)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='8-PSK (simulation)')
    elif k==4:
        plt.semilogy(snr_db,ber,marker='o',label='8-PSK (theory)')

plt.grid(True,which='both')
plt.legend()
plt.xlabel('SNR(Eb/No)  (dB)')
plt.ylabel('ber')
plt.show()
