import numpy as np
import matplotlib.pyplot as plt
import math

# 考慮rayleigh fading的16QAM調變

snr_db = [0]*11
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 100000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)



for k in range(3):
    if k == 0 or k == 1:
        # 定義16QAM的個星座點
        constellation = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j]
        constellation_name = '16QAM'
    elif k == 2:
        # 定義64QAM星座點
        constellation_new = [-7, -5, -3, -1, 1, 3, 5, 7]
        constellation_name = '64QAM'
        constellation = []
        for i in range(len(constellation_new)):
            for j in range(len(constellation_new)):
                constellation += [constellation_new[i] + 1j * constellation_new[j]]

    for i in range(len(snr_db)):

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        # 先將所有可能的星座點能量全部加起來
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)  # 平均一個symbol有Es的能量
        Eb = Es / K  # 平均一個bit有Eb能量
        No = Eb / snr[i]  # 決定No

        if k == 0: # 16 - QAM approximation (rayleigh)
            a = 2 * (1 - 1 / K) / np.log2(K)
            b = 6 * np.log2(K) / (K * K - 1)
            rn = b * snr[i] / 2
            ber[i] = 1 / 2 * a * (1 - np.sqrt(rn / (rn + 1)))
            continue

        error = 0

        for j in range(N):
            b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定送出去的信號
            for m in range(len(constellation)):
                if b <= (m+1)/len(constellation):
                    symbol = constellation[m]
                    break

            #接下來考慮rayleigh fading
            h = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()
            receive = symbol*h

            #接下來加上雜訊
            receive = receive + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

            #接收端利用Maximum Likelihood來detect symbol (缺點是對M - QAM調變系統而言，解調速度過慢)
            #receive = receive/h
            #min_distance = 10**9
            #for m in range(len(constellation)):
            #    if abs(constellation[m] - receive) < min_distance:
            #        detection = constellation[m]
            #        min_distance = abs(constellation[m] - receive)

            # 以下則是採用 decision region的方式來對M - QAM系統做symbol detection(解調速度較快)
            receive = receive/h
            detection = 0
            if constellation_name == '16QAM':
                for m in range(2):
                    if m == 0:  # 先detect symbol的實部
                        if (receive.real < -2):
                            detection += -3
                        elif (receive.real >= -2 and receive.real < 0):
                            detection += -1
                        elif (receive.real >= 0 and receive.real < 2):
                            detection += 1
                        else:
                            detection += 3
                    else:  # 再來detect symbol的虛部
                        if (receive.imag < -2):
                            detection += -3j
                        elif (receive.imag >= -2 and receive.imag < 0):
                            detection += -1j
                        elif (receive.imag >= 0 and receive.imag < 2):
                            detection += 1j
                        else:
                            detection += 3j
            elif constellation_name == '64QAM':
                for m in range(2):
                    if m == 0:  # 先detect symbol的實部
                        if (receive.real < -6):
                            detection += -7
                        elif (receive.real >= -6 and receive.real < -4):
                            detection += -5
                        elif (receive.real >= -4 and receive.real < -2):
                            detection += -3
                        elif (receive.real >= -2 and receive.real < 0):
                            detection += -1
                        elif (receive.real >= 0 and receive.real < 2):
                            detection += 1
                        elif (receive.real >= 2 and receive.real < 4):
                            detection += 3
                        elif (receive.real >= 4 and receive.real < 6):
                            detection += 5
                        else:
                            detection += 7
                    else:  # 再來detect symbol的虛部
                        if (receive.imag < -6):
                            detection += -7j
                        elif (receive.imag >= -6 and receive.imag < -4):
                            detection += -5j
                        elif (receive.imag >= -4 and receive.imag < -2):
                            detection += -3j
                        elif (receive.imag >= -2 and receive.imag < 0):
                            detection += -1j
                        elif (receive.imag >= 0 and receive.imag < 2):
                            detection += 1j
                        elif (receive.imag >= 2 and receive.imag < 4):
                            detection += 3j
                        elif (receive.imag >= 4 and receive.imag < 6):
                            detection += 5j
                        else:
                            detection += 7j

            # 紀錄錯幾個bit
            # 我們這次不找錯幾個symbol，因為在SNR很小時，symbol error rate / 4 不等於 ber error rate
            if detection != symbol :
                if constellation_name == '16QAM':
                    if abs(detection.real-symbol.real) == 2 or abs(detection.real-symbol.real) == 6:
                        error += 1
                    elif abs(detection.real-symbol.real) == 4:
                        error += 2
                    if abs(detection.imag-symbol.imag) == 2 or abs(detection.imag-symbol.imag) == 6:
                        error += 1
                    elif abs(detection.imag-symbol.imag) == 4:
                        error += 2
                elif constellation_name == '64QAM':
                    if abs(detection.real-symbol.real) == 2 or abs(detection.real-symbol.real) == 6 or abs(detection.real-symbol.real) == 14:
                        error += 1
                    elif abs(detection.real-symbol.real) == 4 or abs(detection.real-symbol.real) == 8 or abs(detection.real-symbol.real) == 12:
                        error += 2
                    elif abs(detection.real - symbol.real) == 10:
                        error += 3
                    if abs(detection.imag-symbol.imag) == 2 or abs(detection.imag-symbol.imag) == 6 or abs(detection.imag-symbol.imag) == 14:
                        error += 1
                    elif abs(detection.imag-symbol.imag) == 4 or abs(detection.imag-symbol.imag) == 8 or abs(detection.imag-symbol.imag) == 12:
                        error += 2
                    elif abs(detection.imag - symbol.imag) == 10:
                        error += 3

        ber[i] = error / (K*N)


    if k == 0:
        plt.semilogy(snr_db, ber, marker='o', label='{0} (approximation for rayleigh fading)'.format(constellation_name))
    if k == 1 or k == 2:
        plt.semilogy(snr_db, ber, marker='o', label='{0} (simulation for rayleigh fading)'.format(constellation_name))

plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.ylim(10**(-3),1)
plt.legend()
plt.grid(True,which='both')
plt.show()



