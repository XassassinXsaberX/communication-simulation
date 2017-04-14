import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*10
snr = [0]*10
ber = [0]*10
N = 10000000 #執行N次來找錯誤率
for i in range(10):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [-1,1]

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

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        # 先將所有可能的星座點能量全部加起來
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)  # 平均一個symbol有Es的能量
        Eb = Es / K                       # 平均一個bit有Eb能量

        for j in range(N):
            if k>=0 and k<=2 : #分別為SISO、MRC(1x2)、MRC(1x4)
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for m in range(len(constellation)):
                    if b <= (m + 1) / len(constellation):
                        symbol = constellation[m]
                        break

                No = Eb / snr[i]

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
                h_norm2 = 0
                for l in range(2 ** k):
                    h_norm2 += abs(h[l]) ** 2
                receive_symbol /= h_norm2

                # receive_symbol就是接收端一根天線使用maximum ratio combining 後的結果  ---->  為一純量
                # 接收端利用Maximum Likelihood來detect symbol
                min_distance = 10 ** 9
                for n in range(len(constellation)):
                    if abs(constellation[n] - receive_symbol) < min_distance:
                        detection = constellation[n]
                        min_distance = abs(constellation[n] - receive_symbol)
                        # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                if symbol != detection:
                    error += 1

            elif k==3: #Alamouti(2x1)
                No = 2*Eb / snr[i]              # 因為重複次兩次相同symbol，所以能量變兩倍
                x = [0]*2                       # 要傳送的向量
                X = [[0]*2 for m in range(2)]   # 經過space-time coding 後的傳送symbol matrix

                #接下來決定要送出哪些symbol
                for m in range(2):  # 傳送端一次送出 Nt = 2 個不同symbol
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                    for n in range(len(constellation)):
                        if b <= (n + 1) / len(constellation):
                            x[m] = constellation[n]
                            break

                #接下將要送出去的兩個symbol做space time coding變成X
                X[0][0] = x[0]
                X[1][0] = x[1]
                X[0][1] = -x[1].conjugate()
                X[1][1] = x[0].conjugate()

                #接下來決定通道矩陣
                H = [0]*2
                for m in range(len(H)):
                    H[m] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()

                #接下來決定接收端在時刻1、時刻2收到的兩個純量所構成的向量y
                y = [0]*2
                for m in range(2):
                    for n in range(2):
                        y[m] += H[n] * X[n][m]
                    y[m] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

                #為了decode Alamouti 2x1 的space time coding，我們需要將y向量改變成y_new向量，H也要改變成H_new矩陣
                y_new = [0]*2
                y_new[0] = y[0]
                y_new[1] = y[1].conjugate()

                H_new = [[0]*2 for m in range(2)]
                H_new[0][0] = H[0]
                H_new[0][1] = H[1]
                H_new[1][0] = H[1].conjugate()
                H_new[1][1] = -H[0].conjugate()
                H_new = np.matrix(H_new)

                #可以開始解調
                receive = H_new.getH()*(np.matrix(y_new).transpose()) / (abs(H[0])**2 + abs(H[1])**2)

                for m in range(2):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - receive[m,0]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - receive[m,0])
                            # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if x[m] != detection:
                        error += 1  # error為symbol error 次數



            elif k==4: #Alamouti(2x2)
                No = 2 * Eb / snr[i]             # 因為重複次兩次相同symbol，所以能量變兩倍
                x = [0] * 2                      # 要傳送的向量
                X = [[0] * 2 for m in range(2)]  # 經過space-time coding 後的傳送symbol matrix

                # 接下來決定要送出哪些symbol
                for m in range(2):  # 傳送端一次送出 Nt = 2 個不同symbol
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                    for n in range(len(constellation)):
                        if b <= (n + 1) / len(constellation):
                            x[m] = constellation[n]
                            break

                # 接下將要送出去的兩個symbol做space time coding變成X
                X[0][0] = x[0]
                X[1][0] = x[1]
                X[0][1] = -x[1].conjugate()
                X[1][1] = x[0].conjugate()

                # 接下來決定通道矩陣
                H = [[0]*2 for m in range(2)]
                for m in range(2):
                    for n in range(2):
                        H[m][n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                # 接下來決定接收端在時刻1、時刻2收到的兩個向量所構成的矩陣Y
                # Y = H * X + N
                Y = [[0]*2 for m in range(2)]
                for o in range(2):
                    for m in range(2):
                        for n in range(2):
                            Y[o][m] += H[o][n] * X[n][m]
                        Y[o][m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()


                # 為了decode Alamouti 2x2 的space time coding，我們需要將Y矩陣改變成y_new向量，H也要改變成H_new
                y_new = [0]*4
                y_new[0] = Y[0][0]
                y_new[1] = Y[1][0]
                y_new[2] = Y[0][1].conjugate()
                y_new[3] = Y[1][1].conjugate()

                H_new = [[0]*2 for m in range(4)]
                H_new[0][0] = H[0][0]
                H_new[0][1] = H[0][1]
                H_new[1][0] = H[1][0]
                H_new[1][1] = H[1][1]
                H_new[2][0] = H[0][1].conjugate()
                H_new[2][1] = -H[0][0].conjugate()
                H_new[3][0] = H[1][1].conjugate()
                H_new[3][1] = -H[1][0].conjugate()
                H_new = np.matrix(H_new)


                # 可以開始解調
                receive = H_new.getH() * (np.matrix(y_new).transpose()) / (abs(H[0][0]) ** 2 + abs(H[0][1]) ** 2 + abs(H[1][0]) ** 2 + abs(H[1][1]) ** 2)

                for m in range(2):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - receive[m, 0]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - receive[m, 0])
                            # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if x[m] != detection:
                        error += 1  # error為symbol error 次數


        if k>=0 and k<=2:
            ber[i] = error / (K*N)  # 分母乘上K是因為一個symbol含有K個bit
        else:
            ber[i] = error/ (2*K*N) # 分母乘上K是因為一個symbol含有K個bit、分母乘上2是因為space time code含有兩個symbol


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
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="SISO theory for BPSK")
    elif k==6:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) theory for BPSK")
    elif k==7:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2) theory for BPSK")
plt.ylabel("BER")
plt.xlabel("Eb/No , dB")
plt.grid(True,which='both')
plt.legend()
plt.show()