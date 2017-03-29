#BPSK  的SISO、MRC(2x1)、MRC(4x1) 錯誤率
#(2x1)代表傳送端有兩根天線，接收端有一根天線
import numpy as np
import matplotlib.pyplot as plt


snr_db = [0]*10
snr = [0]*10
ber = [0]*10
N = 10000000 #執行N次來找錯誤率
for i in range(10):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(6):#總共有SISO、MRC(1x2)、MRC(1x4)、SISO(theory)、MRC(1x2) (theory)、MRC(2x1)
    for i in range(len(snr)):
        if k == 3:  # SISO(theory)
            ber[i] = 1/2-1/2*np.power(1+1/snr[i],-1/2)
        elif k==4:  # MRC(1x2) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
        elif k==5:  #  MRC(2x1)  --->   模擬結果同SISO
            error = 0
            for j in range(N):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol = 1
                else:
                    symbol = -1
                # SNR = Eb / No
                # Eb = 1   so   No = 1 / SNR
                # noise的variance = No/2
                No = 2 / snr[i]   #因為重複誦兩次symbol，所以能量變兩倍
                h = [0] * 2
                receive = 0
                for l in range(2):
                    h[l] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()  # 產生 rayleigh 分布的通道模型
                    receive += symbol * h[l]
                receive += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

                # 接下來使用match filter
                receive_symbol = receive * (h[0]+h[1]).conjugate()
                # receive_symbol就是接收端一根天線使用maximum ratio combining 後的結果  ---->  為一純量
                if abs(receive_symbol - 1) < abs(receive_symbol + 1):
                    receive_symbol = 1
                else:
                    receive_symbol = -1
                if symbol != receive_symbol:
                    error += 1
            ber[i] = error / N
        else:   # 1x1 SISO通道、1x2 MISO通道、1x4 MISO通道
            error = 0
            for j in range(N):
                b = np.random.random() #產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol = 1
                else:
                    symbol = -1
                # SNR = Eb / No
                # Eb = 1   so   No = 1 / SNR
                # noise的variance = No/2
                No = 1 / snr[i]
                h = [0]*(2**k)
                receive = [0]*(2**k)
                # 若k=0 代表 1x1 SISO通道，所以只要產生一個通道即可
                # 若k=1 代表 1x2 MISO通道，所以要產生兩個通道
                # 若k=2 代表 1x4 MISO通道，所以要產生四個通道
                for l in range(2**k):
                    h[l] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()  #產生 rayleigh 分布的通道模型
                    receive[l] = symbol*h[l] + np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                receive_symbol = 0
                #接下來使用match filter
                for l in range(2**k):
                    receive_symbol += receive[l]*(h[l].conjugate())
                #receive_symbol就是接收端一根天線使用maximum ratio combining 後的結果  ---->  為一純量
                if abs(receive_symbol-1) < abs(receive_symbol+1) :
                    receive_symbol = 1
                else:
                    receive_symbol = -1
                if symbol != receive_symbol:
                    error += 1
            ber[i] = error/N
    if k==0:
        plt.semilogy(snr_db, ber,marker='o', linestyle='-', label="SISO")
    elif k>=1 and k<=2:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x{0})".format(2**k))
    elif k == 3:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="SISO(theory)")
    elif k == 4:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2) (theory)")
    elif k == 5:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(2x1)")

plt.ylabel("BER")
plt.xlabel("SNR(db)")
plt.grid(True,which='both')
plt.legend()
plt.show()