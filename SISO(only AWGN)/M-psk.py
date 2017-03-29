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
                No = 1/snr[i]
                symbol = np.random.random()
                if symbol > 0.5:
                    symbol = 1
                else:
                    symbol=-1
                receive_symbol = symbol + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                if abs(receive_symbol-1) < abs(receive_symbol+1):
                    receive_symbol = 1
                else:
                    receive_symbol = -1
                if receive_symbol != symbol:
                    error +=1

            elif k == 2 :#qpsk
                # SNR = Eb / No
                #若星座點為1+1j，則Es = 2，Eb = 1，故SNR = 1 / No
                No = 1/snr[i]

                #決定星座圖
                map = [0]*4
                map[0] = 1+1j
                map[1] = -1+1j
                map[2] = -1-1j
                map[3] = 1-1j

                #決定送出哪星座點symbol
                symbol = np.random.random() # 產生一個 (0,1) uniform 分布的隨機變數
                if symbol < 1/4:
                    symbol = map[0]
                elif symbol< 2/4:
                    symbol = map[1]
                elif symbol< 3/4:
                    symbol = map[2]
                else:
                    symbol = map[3]

                #送出的symbol加上AWGN雜訊
                receive_symbol = symbol + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

                #判斷收到的信號梨哪個星座點最近，用此來進行detection
                m = 10000000
                for n in range(4):
                    if abs(receive_symbol-map[n]) < m:
                        m = abs(receive_symbol-map[n])
                        receive = map[n]
                if receive != symbol:
                    error +=1

            elif k == 3 :#8-psk
                # SNR = Eb / No
                # 若Es = 1，因為Es = 3*Eb。所以Eb = 1/3，故SNR = (1/3) / No
                No = (1/3)/snr[i]
                map = [0]*8
                for m in range(len(map)):
                    map[m] = np.cos(2*np.pi/8*(m-1))-1j*np.sin(2*np.pi/8*(m-1))

                symbol = np.random.random()
                for m in range(len(map)):
                    if symbol <= (m+1)/8:
                        symbol = map[m]
                        break

                receive_symbol = symbol + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                m = 10000000
                for n in range(8):
                    if abs(receive_symbol-map[n]) < m:
                        m = abs(receive_symbol-map[n])
                        receive = map[n]
                if receive != symbol:
                    error +=1

        if k == 0:
            ber[i] = error/N
        elif k == 2: #因為qpsk 的 bit error rate 大約為 (symbol error rate / 2)
            ber[i] = error/N/2
        elif k == 3: # 因為8-psk 的 bit error rate 大約為 (symbol error rate / 3)
            ber[i] = error/N/3
    if k==0:
        plt.semilogy(snr_db,ber,label='bpsk (simulation)')
    elif k==1:
        plt.semilogy(snr_db,ber,label='bpsk (theory)')
    elif k==2:
        plt.semilogy(snr_db,ber,label='qpsk (simulation)')
    elif k==3:
        plt.semilogy(snr_db,ber,label='8-psk (simulation)')
    elif k==4:
        plt.semilogy(snr_db,ber,label='8-psk (theory)')

plt.grid(True,which='both')
plt.legend()
plt.xlabel('SNR(Eb/No)  (dB)')
plt.ylabel('ber')
plt.show()
