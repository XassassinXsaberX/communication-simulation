import numpy as np
import matplotlib.pyplot as plt
import math

# reference : http://www.dsplog.com/2008/06/05/16qam-bit-error-gray-mapping/

snr_db = [0]*12
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = i
    snr[i] = np.power(10,snr_db[i]/10)

# 先決定16QAM的16個星座點
constellation =  [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]

for k in range(3):
    for i in range(len(snr_db)):
        if k == 0: #16QAM theory
            # theroy 1
            ber[i] = 2*(np.sqrt(16)-1)/np.sqrt(16)/4*math.erfc(np.sqrt((4*snr[i])/10))

            # theroy 2
            #ber[i] = 3/4*(1/2)*math.erfc(np.sqrt(snr[i]*4/10))
            #ber[i] += 1/2*(1/2)*math.erfc(3*np.sqrt(snr[i]*4/10))
            #ber[i] -= 1/4*(1/2)*math.erfc(5*np.sqrt(snr[i]*4/10))
            continue
        elif k == 1:#16-psk theory
            ber[i] = 1/4*math.erfc(np.sqrt(4*snr[i])*np.sin(np.pi/16))
            continue
        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        # 先將所有可能的星座點能量全部加起來
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)      # 平均一個symbol有Es的能量
        Eb = Es / K                           # 平均一個bit有Eb能量
        No = Eb / snr[i]                      # 決定No

        for j in range(N):
            b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定送出去的信號
            for m in range(len(constellation)):
                if b <= (m+1)/len(constellation):
                    symbol = constellation[m]
                    break

            #接下來加上雜訊
            receive = symbol + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

            #接收端利用Maximum Likelihood來detect symbol
            min_distance = 10**9
            for m in range(len(constellation)):
                if abs(constellation[m] - receive) < min_distance:
                    detection = constellation[m]
                    min_distance = abs(constellation[m] - receive)

            # 紀錄錯幾個bit
            # 我們這次不找錯幾個symbol，因為在SNR很小時，symbol error rate / 4 不等於 ber error rate
            if detection != symbol :
                if abs(detection.real-symbol.real) == 2 or abs(detection.real-symbol.real) == 6:
                    error += 1
                elif abs(detection.real-symbol.real) == 4:
                    error += 2
                if abs(detection.imag-symbol.imag) == 2 or abs(detection.imag-symbol.imag) == 6:
                    error += 1
                elif abs(detection.imag-symbol.imag) == 4:
                    error += 2


        ber[i] = error / (K*N) #因為平均一個symbol含有K個bit


    if k == 0:
        plt.semilogy(snr_db, ber, marker='o', label='16QAM (theory)')
    if k == 1:
        plt.semilogy(snr_db, ber, marker='o', label='16-psk (theory)')
    if k == 2:
        plt.semilogy(snr_db, ber, marker='o', label='16QAM (simulation)')

plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.legend()
plt.grid(True,which='both')
plt.show()



