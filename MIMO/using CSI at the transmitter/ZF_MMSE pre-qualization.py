import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*12
snr = [0]*12
ber = [0]*12
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [ -1, 1 ]  #定義星座點的集合


for k in range(6):
    for i in range(len(snr_db)):
        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)  # 平均一個symbol有Es的能量
        Eb = Es / K                       # 平均一個bit有Eb能量
        # 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大

        if k == 0:# MRC(1x2) for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k == 1:# SISO for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue

        elif k == 2 or k == 3:#pre-equalization
            # 注意做pre-equalization時 Nt >= Nr
            Nr = 2             # 接收端天線數
            Nt = 3             # 傳送端天線數
            # 這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
            H = [[0j] * Nt for i in range(Nr)]
            H = np.matrix(H)
            symbol = [0] * Nr  # 雖然接收端有Nt根天線，但實際上一次只會送Nr個，且Nr < Nt
            y = [0] * Nr       # 接收端的向量
            No = Eb * Nt/Nr / snr[i]

        elif k == 4 or k==5:#equalization
            # 注意做equalization時 Nt <= Nr
            Nr = 3             # 接收端天線數
            Nt = 2             # 傳送端天線數
            # 這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
            H = [[0j] * Nt for i in range(Nr)]
            H = np.matrix(H)
            symbol = [0] * Nt  # 因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
            y = [0] * Nr       # 接收端的向量
            No = Eb / snr[i]


        for j in range(N):
            # 決定要送哪些symbol
            for m in range(len(symbol)):  # 傳送端一次送出Nr個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break

            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            if k == 2 or k == 3:# pre-equalization
                # 首先要決定weight matrix W
                if k == 2:    # ZF pre-equalization
                    W = H.getH() * (H*H.getH()).I
                elif k == 3:  # MMSE pre-equalization
                    W = H.getH() * (H*H.getH() + 1/snr[i]*np.identity(Nr)).I
                beta = np.sqrt(Nt / complex((W * (W.getH())).trace()))
                W = beta * W

                # 接下來將要送出去symbol vector先乘上W得到codeword向量
                codeword = [0]*Nt
                for m in range(Nt):
                    for n in range(Nr):
                        codeword[m] += W[m,n] * symbol[n]

                # 接下來送出codeword向量，數學模型為 H(matrix)*codeword(vector) + noise(vector)
                # 接下來決定接收端收到的向量y (共有Nr的元素)
                for m in range(Nr):
                    y[m] = 0
                for m in range(Nr):
                    for n in range(Nt):
                        y[m] += H[m, n] * codeword[n]
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

                #接收端收到y向量後先除以beta後，才可以直接解調
                for m in range(Nr):
                    y[m] /= beta

                for m in range(Nr):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - y[m]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - y[m])
                            # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if symbol[m] != detection:
                        error += 1  # error為symbol error 次數

            elif k == 4 or k == 5: # equalization
                # 決定要送哪些symbol
                for m in range(Nt):  # 傳送端一次送出Nt個不同symbol
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                    for n in range(len(constellation)):
                        if b <= (n + 1) / len(constellation):
                            symbol[m] = constellation[n]
                            break

                # 先決定MIMO的通道矩陣
                for m in range(Nr):
                    for n in range(Nt):
                        H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                # 接下來決定接收端收到的向量y (共有Nr的元素)
                for m in range(Nr):
                    y[m] = 0
                for m in range(Nr):
                    for n in range(Nt):
                        y[m] += H[m, n] * symbol[n]
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
                if k == 4:
                    # 決定ZF 的weight matrix
                    W = ((H.getH() * H ).I) * H.getH()  # W為 Nt x Nr 矩陣
                elif k == 5:
                    # 決定MMSE 的weight matrix
                    W = ((H.getH() * H + 1 / snr[i] * np.identity(Nt)).I) * H.getH()  # W為 Nt x Nr 矩陣

                # 接收端做equalization : receive向量 = W矩陣 * y向量
                receive = [0] * Nt
                for m in range(Nt):
                    for n in range(Nr):
                        receive[m] += W[m, n] * y[n]

                for m in range(Nt):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - receive[m]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - receive[m])
                    # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if symbol[m] != detection:
                        error += 1  # error為symbol error 次數

        if k == 2 or k == 3:
            ber[i] = error / (K*Nr*N)  # 除以K是因為一個symbol有K個bit、分母乘上Nr是因為傳送端一次送Nr個元素，而不是Nt個
        elif k == 4 or k == 5:
            ber[i] = error / (K*Nt*N)  # 除以K是因為一個symbol有K個bit、分母乘上Nr是因為傳送端一次送Nt個元素

    if k == 0 :
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1x2) for BPSK (theory)')
    elif k == 1:
        plt.semilogy(snr_db,ber,marker='o',label='SISO for BPSK (theory)')
    elif k == 2:
        plt.semilogy(snr_db,ber,marker='o',label='pre-ZF equalization Nt={0}, Nr={1}'.format(Nt,Nr))
    elif k == 3:
        plt.semilogy(snr_db, ber, marker='o', label='pre-MMSE equalization Nt={0}, Nr={1}'.format(Nt,Nr))
    elif k == 4:
        plt.semilogy(snr_db, ber, marker='o', label='ZF equalization Nt={0}, Nr={1}'.format(Nt,Nr))
    elif k == 5:
        plt.semilogy(snr_db, ber, marker='o', label='MMSE equalization Nt={0}, Nr={1}'.format(Nt,Nr))

plt.legend()
plt.ylabel('ber')
plt.xlabel('Eb/No , dB')
plt.grid(True,which='both')
plt.show()
