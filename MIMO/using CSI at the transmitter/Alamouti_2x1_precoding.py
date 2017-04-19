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

#接下來要產生codebook
column_index = [0,1]
rotation_vector = [1,7,52,56]
codebook_size = 64             #codebook中有多少種precoding matrix
N_m = len(column_index)        #codeword length (你也可以看成是data stream 數目)
N_Nt = 4                       #發送端天線數量
codebook = []                  #codebook裡面會存放許多precoding matrix

W_DFT = [[0]*N_Nt for i in range(N_Nt)]
for i in range(N_Nt):
    for j in range(N_Nt):
        W_DFT[i][j] = 1/np.sqrt(N_Nt)*np.exp(1j*2*np.pi*i*j/N_Nt)

#藉由剛剛找的W_DFT可以找出我們的第一個precoding matrix W
W = [[0]*N_m for i in range(N_Nt)]
for i in range(N_m):
    for j in range(N_Nt):
        W[j][i] = W_DFT[j][column_index[i]]
W = np.matrix(W)

#定義對角矩陣為theta
theta = [[0]*N_Nt for i in range(N_Nt)]
for i in range(N_Nt):
    theta[i][i] = np.exp(-1j*2*rotation_vector[i]/N_Nt)
theta = np.matrix(theta)

#接下來開始製造其他precoding matrix
codebook.append(W)
for i in range(1,codebook_size,1):
    W = theta * W
    codebook.append(W)
#codebook 製造完成

for k in range(4):#總共有Alamouti(2x1) theory、MRC(1x2) theory、Alamouti(2x1)、Alamouti(2x1) precoding
    for i in range(len(snr)):
        if k == 0:#找出Alamouti(2x1) theory
            ber[i] = 1/2-1/2*np.power(1+2/snr[i],-1/2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        elif k == 1:#MRC(2x1) (theory)
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
            if k==2 : #Alamouti(2x1)
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
                H = [0] * 2
                for m in range(len(H)):
                    H[m] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                # 接下來決定接收端在時刻1、時刻2收到的兩個純量所構成的向量y
                y = [0] * 2
                for m in range(2):
                    for n in range(2):
                        y[m] += H[n] * X[n][m]
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

                # 為了decode Alamouti 2x1 的space time coding，我們需要將y向量改變成y_new向量，H也要改變成H_new矩陣
                y_new = [0] * 2
                y_new[0] = y[0]
                y_new[1] = y[1].conjugate()

                H_new = [[0] * 2 for m in range(2)]
                H_new[0][0] = H[0]
                H_new[0][1] = H[1]
                H_new[1][0] = H[1].conjugate()
                H_new[1][1] = -H[0].conjugate()
                H_new = np.matrix(H_new)

                # 可以開始解調
                receive = H_new.getH() * (np.matrix(y_new).transpose()) / (abs(H[0]) ** 2 + abs(H[1]) ** 2)

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

            elif k==3: #Alamouti(2x1) precoding  --->  注意他是用4根天線來發送的...
                No = 2 * Eb / snr[i]            # 經過數學推導後我們會發現採用該方案時，平均一個symbol送了2Es，所以平均一個bit送了2Eb
                x = [0] * 2                     # 要傳送的向量
                X = [[0] * 2 for m in range(2)] # 經過space-time coding 後的傳送symbol matrix

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
                H = [0] * N_Nt
                for m in range(len(H)):
                    H[m] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                # 在來要從codebook中找出最好的precoding matrix
                max_norm = 0
                best_index = -1
                for m in range(codebook_size):
                    # 先算出H*W
                    HW = [0] * N_m
                    for o in range(N_m):
                        for p in range(N_Nt):
                            HW[o] += H[p] * codebook[m][p, o]
                    # 再算出 H*W 的norm 平方
                    norm = 0
                    for n in range(N_m):
                        norm += abs(HW[n]) ** 2
                        # 目標是找出最佳的precoding matrix W，使得H*W的norm平方有最大值
                    if norm > max_norm:
                        max_norm = norm
                        best_index = m

                # 找出最佳的precoding matrix了，接下來H*W得到新的channel matrix
                He = np.matrix(H) * codebook[best_index]

                # 接下來決定接收端在時刻1、時刻2收到的兩個純量所構成的向量y
                y = [0] * 2
                for m in range(2):
                    for n in range(2):
                        y[m] += He[0, n] * X[n][m]
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

                # 為了decode Alamouti 2x1 的space time coding，我們需要將y向量改變成y_new向量，H也要改變成H_new矩陣
                y_new = [0] * 2
                y_new[0] = y[0]
                y_new[1] = y[1].conjugate()

                H_new = [[0] * 2 for m in range(2)]
                H_new[0][0] = He[0, 0]
                H_new[0][1] = He[0, 1]
                H_new[1][0] = He[0, 1].conjugate()
                H_new[1][1] = -He[0, 0].conjugate()
                H_new = np.matrix(H_new)

                # 可以開始解調
                receive = H_new.getH() * (np.matrix(y_new).transpose()) / (abs(H[0]) ** 2 + abs(H[1]) ** 2)

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

        ber[i] = error/ (2*K*N) # 分母乘上K是因為一個symbol含有K個bit、分母乘上2是因為space time coding 會送兩個symbol


    if k==0:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) theory for BPSK")
    elif k==1:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2) theory for BPSK")
    elif k==2:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1)")
    elif k==3:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) precoding")



plt.ylabel("BER")
plt.xlabel("Eb/No , dB")
plt.grid(True,which='both')
plt.legend()
plt.show()