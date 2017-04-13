import numpy as np
import matplotlib.pyplot as plt
import math

snr_db = [0]*12
snr = [0]*12
ber = [0]*12
Nt = 2 #傳送端天線數
Nr = 2 #接收端天線數
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [ -1, 1 ]  #定義星座點的集合

#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = [0]*Nt #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = [0]*Nr      #接收端的向量

for k in range(5):
    for i in range(len(snr)):
        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)    # 平均一個symbol有Es的能量
        Eb = Es / K                         # 平均一個bit有Eb能量
        # 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大
        No = Eb / snr[i]                    # 最後決定No

        if k==2:# MRC(1x2) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k==3:# SISO(BPSK) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue
        elif k==4:# SISO(BPSK) (theory)
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue

        for j in range(N):
            #決定要送哪些symbol
            for m in range(Nt): #傳送端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break

            #先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m,n] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()

            #接下來決定接收端收到的向量y (共有Nr的元素)
            for m in range(Nr):
                y[m] = 0
            for m in range(Nr):
                for n in range(Nt):
                    y[m] += H[m,n]*symbol[n]
                y[m] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

            if k==0:#執行ZF detection
                #決定ZE 的weight matrix
                W = ((H.getH()*H)**(-1))*H.getH()  #W為 Nt x Nr 矩陣
            elif k == 1:  # 執行MMSE detection
                # 決定MMSE 的weight matrix
                W = ((H.getH() * H + 1 / snr[i] * np.identity(Nt)) ** (-1)) * H.getH()  # W為 Nt x Nr 矩陣

            # receive向量 = W矩陣 * y向量
            receive = [0]*Nt
            for m in range(Nt):
                for n in range(Nr):
                    receive[m] += W[m,n]*y[n]

            for m in range(Nt):
                # 接收端利用Maximum Likelihood來detect symbol
                min_distance = 10 ** 9
                for n in range(len(constellation)):
                    if abs(constellation[n] - receive[m]) < min_distance:
                        detection = constellation[n]
                        min_distance = abs(constellation[n] - receive[m])
                # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                if symbol[m] != detection:
                    error += 1 # error為symbol error 次數


        ber[i] = error/(K*Nt*N) #除以K是因為一個symbol有K個bit

    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='ZF')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='MMSE')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1x2) for BPSK (theory)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='SISO for BPSK (theory-formula1)')
    elif k==4:
        plt.semilogy(snr_db,ber,marker='o',label='SISO for BPSK (theory-formula2)')
plt.legend()
plt.ylabel('ber')
plt.xlabel('snr (Eb/No) dB')
plt.grid(True,which='both')
plt.show()