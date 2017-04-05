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
        # 假設送出去的symbol有 1+j、1+3j、3+j、3+3j、-1+j、-1+3j、-3+j、-3+3j、-1-j、-1-3j、-3-j、-3-3j、1-j、1-3j、3-j、3-3j
        # 能量分別為                      2  、10   、10 、18   、  2  、  10  、  10 、  18  、  2  、  10 、 10 、  18  、 2 、 10 、 10、 18
        # 平均一個symbol 能量為 ( 2 + 10 + 10 + 18) * 4 / 16 = 10
        # 其實若QAM為方形(M = L*L)，則 Es  = 2 / 3 * Eo * (M-1)-------------------------(請見我的筆記推導)
        # 16QAM : Eo = 1(因為是用1+j)，M = 16 = 4*4，所以可推出 Es = 10
        # 而16QAM每個symbol含有4bit，所以Eb = 10 / 4
        # snr = Eb / No，故No = 10 / 4 / snr
        No = 10 / 4 / snr[i]
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


        ber[i] = error / (4*N)


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



