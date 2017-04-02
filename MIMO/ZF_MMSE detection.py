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


#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = [0]*Nt #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = [0]*Nr  #接收端的向量

for k in range(5):
    for i in range(len(snr)):
        error = 0
        No = 1/snr[i]  #每個symbol只送一次能量
        #已知 SNR = Eb / No
        #令symbol 能量 Es =1 ，因為一次只送一個symbol所以Eb = Es = 1
        #所以 No = 1 / SNR
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
            #決定要送哪些symbol (採用BPSK調變)
            for m in range(Nt): #接收端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol[m] = 1
                else:
                    symbol[m] = -1

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
                receive = [0]*Nt
                for m in range(Nt):
                    for n in range(Nr):
                        receive[m] += W[m,n]*y[n]

                for m in range(Nt):
                    if abs(receive[m]-1) < abs(receive[m]+1):
                        receive_symbol = 1
                    else:
                        receive_symbol = -1
                    if symbol[m] != receive_symbol:
                        error += 1
            elif k==1:#執行MMSE detection
                #決定MMSE 的weight matrix
                W = ((H.getH()*H + 1/snr[i]*np.identity(Nt))**(-1))*H.getH()  #W為 Nt x Nr 矩陣
                receive = [0]*Nt
                for m in range(Nt):
                    for n in range(Nr):
                        receive[m] += W[m,n]*y[n]

                for m in range(Nt):
                    if abs(receive[m]-1) < abs(receive[m]+1):
                        receive_symbol = 1
                    else:
                        receive_symbol = -1
                    if symbol[m] != receive_symbol:
                        error += 1

        ber[i] = error/(Nt*N)
    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='ZF')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='MMSE')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1X2) (theory)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='SISO(BPSK) (theory-formula1)')
    elif k==4:
        plt.semilogy(snr_db,ber,marker='o',label='SISO(BPSK) (theory-formula2)')
plt.legend()
plt.ylabel('ber')
plt.xlabel('snr (Eb/No) dB')
plt.grid(True,which='both')
plt.show()