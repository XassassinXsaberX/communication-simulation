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

for k in range(4):#總共有Alamouti(2x1) BPSK theory、MRC for BPSK theory (1x2)、Alamouti(2x1)、Alamouti(2x1) antenna selection
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

            if k==2:  # Alamouti(2x1)
                No = 2 * Eb / snr[i]  # 因為重複次兩次相同symbol，所以能量變兩倍
                #接下來決定通道矩陣
                H = [0]*2
                for m in range(len(H)):
                    H[m] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()

            elif k==3: # Alamouri(2x1) antenna selection
                No = 2 * Eb / snr[i]  # 因為重複次兩次相同symbol，所以能量變兩倍
                Nt = 4
                Nr = 1
                # 接下來決定通道矩陣
                H = [0] * Nt  #因為傳送端有Nt根天線
                for m in range(len(H)):
                    H[m] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                # 接下來要從Nt根傳送天線中選兩根出來傳送data
                # 若有4根傳送天線，則H = [ h1 , h2 , h3 , h4 ]  其中h1,h2,h3,h4 都是column vector，都各自代表一根傳送端天線
                # 我們選前兩個有較大norm平方的column vector
                # 若為h2 和 h3 則代表我們要用第2, 3 根天線來送data
                H_norm_2 = [[0,0] for m in range(Nt)]           #第一個數字代表每個column vector的norm平方，第二個數字紀錄column的index(待會排序完會用到)
                for m in range(Nt):
                    H_norm_2[m][1] = m
                for m in range(Nt):
                    H_norm_2[m][0] += abs(H[m])**2
                H_norm_2.sort(reverse=True,key = lambda cust: cust[0])

                # 排序完後，我們選norm平方較大的前兩個column vector組成新的channel vector (也就是選那兩個天線來送data)
                index = [H_norm_2[0][1],H_norm_2[1][1]]
                index.sort() # 若h3的norm平方 > h2的norm平方，則新的矩陣應為[ h2 , h3 ]而不是[ h3 , h2 ]，所以我們才要對前兩個index進行由小大到的排序
                H_new = [0]*2
                for m in range(2):
                    H_new[m] = H[index[m]]
                H = H_new


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

        ber[i] = error/ (2*K*N) # 分母乘上K是因為一個symbol含有K個bit、分母乘上2是因為space time code含有兩個symbol


    if k==0:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) theory for BPSK")
    elif k==1:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="MRC(1x2) theory for BPSK")
    elif k==2:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1)")
    elif k==3:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="Alamouti(2x1) antenna selection")

plt.ylabel("BER")
plt.xlabel("Eb/No , dB")
plt.grid(True,which='both')
plt.legend()
plt.show()