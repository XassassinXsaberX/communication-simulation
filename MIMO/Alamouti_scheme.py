import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*10
snr = [0]*10
ber = [0]*10
N = 10000000 #執行N次來找錯誤率
for i in range(10):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(8):#總共有SISO、SISO theory、Alamouti(2x1)、Alamouti(2x1) theory、Alamouti(2x2)、MRC(1x2)、MRC(1x2) theory、MRC(1x4)
    for i in range(len(snr)):
        if k == 5:  # SISO(theory)
            ber[i] = 1/2-1/2*np.power(1+1/snr[i],-1/2)
            continue
        elif k == 6:#找出Alamouti(2x1) theory
            ber[i] = 1/2-1/2*np.power(1+2/snr[i],-1/2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        elif k == 7:#MRC(2x1) (theory)
            ber[i] = 1/2-1/2*np.power(1+1/snr[i],-1/2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        error = 0
        for j in range(N):
            # SNR = Eb / No
            # Eb = 1   so   No = 1 / SNR
            # noise的variance = No/2
            if k>=0 and k<=2: #SISO、MRC(1x2)、MRC(1x4)
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol = 1
                else:
                    symbol = -1
                # SNR = Eb / No
                # Eb = 1   so   No = 1 / SNR
                # noise的variance = No/2
                No = 1 / snr[i]
                h = [0] * (2 ** k)
                receive = [0] * (2 ** k)
                # 若k=0 代表 1x1 SISO通道，所以只要產生一個通道即可
                # 若k=1 代表 1x2 MISO通道，所以要產生兩個通道
                # 若k=2 代表 1x4 MISO通道，所以要產生四個通道
                for l in range(2 ** k):
                    h[l] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()  # 產生 rayleigh 分布的通道模型
                    receive[l] = symbol * h[l] + np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
                receive_symbol = 0
                # 接下來使用match filter
                for l in range(2 ** k):
                    receive_symbol += receive[l] * (h[l].conjugate())
                # receive_symbol就是接收端一根天線使用maximum ratio combining 後的結果  ---->  為一純量
                if abs(receive_symbol - 1) < abs(receive_symbol + 1):
                    receive_symbol = 1
                else:
                    receive_symbol = -1
                if symbol != receive_symbol:
                    error += 1

            elif k==3: #Alamouti(2x1)
                No = 2 / snr[i]          # 因為重複次兩次相同symbol，所以能量變兩倍
                #已知 SNR = Eb / No
                #令symbol 能量 Es =1 ，因為一次重複送兩個symbol所以Eb = 2Es = 2
                #所以 No = 2 / SNR
                b1 = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數來決定送哪個bit
                b2 = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數來決定送哪個bit
                if b1 >= 0.5:
                    symbol1 = 1+0j
                else:
                    symbol1 = -1+0j
                if b2 >= 0.5:
                    symbol2 = 1+0j
                else:
                    symbol2 = -1+0j
                x = [[symbol1],[symbol2]]
                h = [0]*2
                h[0] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()
                h[1] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()
                H = [[0]*2 for i in range(2)]
                H[0][0] = h[0]
                H[0][1] = h[1]
                H[1][0] = h[1].conjugate()
                H[1][1] = -h[0].conjugate()
                y = [[0],[0]]
                for m in range(2):
                    y[m][0] = 0
                    for n in range(2):
                        y[m][0] += H[m][n]*x[n][0]
                y[0][0] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                y[1][0] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                H = np.matrix(H)
                y = np.matrix(y)
                x1 = (H.getH()*y)[0][0]/(abs(h[0])*abs(h[0]) + abs(h[1])*abs(h[1]))
                x2 = (H.getH()*y)[1][0]/(abs(h[0])*abs(h[0]) + abs(h[1])*abs(h[1]))
                if abs(x1-1) < abs(x1+1):
                    x1 = 1
                else:
                    x1 = -1
                if abs(x2-1) < abs(x2+1):
                    x2 = 1
                else:
                    x2 = -1
                if x1 != symbol1:
                    error += 1
                if x2 != symbol2:
                    error += 1

            elif k==4: #Alamouti(2x2)
                No = 2 / snr[i]          # 因為重複次兩次相同symbol，所以能量變兩倍
                b1 = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數來決定送哪個bit
                b2 = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數來決定送哪個bit
                if b1 >= 0.5:
                    symbol1 = 1+0j
                else:
                    symbol1 = -1+0j
                if b2 >= 0.5:
                    symbol2 = 1+0j
                else:
                    symbol2 = -1+0j
                x = [[symbol1],[symbol2]]
                h = [0]*4
                for l in range(4):
                    h[l] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()
                H = [[0]*2 for l in range(4)]
                H[0][0] = h[0]
                H[0][1] = h[1]
                H[1][0] = h[2]
                H[1][1] = h[3]
                H[2][0] = h[1].conjugate()
                H[2][1] = -h[0].conjugate()
                H[3][0] = h[3].conjugate()
                H[3][1] = -h[2].conjugate()
                y = [[0] for l in range(4)]
                for m in range(4):
                    y[m][0] = 0
                    for n in range(2):
                        y[m][0] += H[m][n]*x[n][0]
                for l in range(4):
                    y[l][0] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                H = np.matrix(H)
                y = np.matrix(y)
                x1 = (H.getH()*y)[0][0]/(abs(h[0])*abs(h[0]) + abs(h[1])*abs(h[1]) + abs(h[2])*abs(h[2]) + abs(h[3])*abs(h[3]))
                x2 = (H.getH()*y)[1][0]/(abs(h[0])*abs(h[0]) + abs(h[1])*abs(h[1]) + abs(h[2])*abs(h[2]) + abs(h[3])*abs(h[3]))

                if abs(x1-1) < abs(x1+1):
                    x1 = 1
                else:
                    x1 = -1
                if abs(x2-1) < abs(x2+1):
                    x2 = 1
                else:
                    x2 = -1
                if x1 != symbol1:
                    error += 1
                if x2 != symbol2:
                    error += 1

        if k>=0 and k<=2:
            ber[i] = error/N
        else:
            ber[i] = error/(2*N)


    if k==0:
        plt.semilogy(snr_db,ber,marker='o',linestyle='-',label="SISO")
    elif k==1:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2)")
    elif k==2:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x4)")
    elif k==3:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1)")
    elif k==4:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x2)")
    elif k==5:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="SISO theory")
    elif k==6:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) theory")
    elif k==7:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2) theory")
plt.ylabel("BER")
plt.xlabel("SNR(db)")
plt.grid(True,which='both')
plt.legend()
plt.show()