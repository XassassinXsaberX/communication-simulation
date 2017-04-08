import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*10
snr = [0]*10
ber = [0]*10
N = 1000000  #執行N次來找錯誤率
Nt = 2       #傳送端天線數
Nr = 2       #接收端天線數
for i in range(10):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = [0]*Nt #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = [0]*Nr  #接收端的向量


for k in range(4):
    for i in range(len(snr)):
        error = 0
        total = 0
        if k == 0:  # SISO theory
            ber[i] = 1/2 - 1/2*np.power(1+1/snr[i], -1/2)
            continue
        elif k ==1 : # MRC (1x2) theroy
            ber[i] = 1/2 - 1/2*np.power(1+1/snr[i], -1/2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        for j in range(N):
            if k==2 :
                # 採用BPSK調變
                constellation = [1,-1]
            elif k==3 :
                # 採用QPSK調變
                constellation = [1+1j, 1-1j, -1+1j, -1-1j]

            K = int(np.log2(len(constellation))) #代表一個symbol含有K個bit
            #接下來要算平均一個symbol有多少能量
            energy = 0
            for m in range(len(constellation)):
                energy += abs(constellation[m])**2
            Es = energy / len(constellation)  #平均一個symbol有Es的能量
            Eb = Es / K                       #平均一個bit有Eb能量
            #因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大
            No = Eb / snr[i]

            for m in range(Nt):  # 接收端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n+1)/len(constellation):
                        symbol[m] = constellation[n]

            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            #接下來決定接收端收到的向量
            for m in range(len(y)):
                y[m] = 0
            for m in range(Nr):
                for n in range(Nt):
                    y[m] += H[m,n]*symbol[n]
                y[m] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

            # 接下來進行ML detection
            # 找出傳送端所有可能送出的向量經過通道後的結果，並和目前收到的接收向量進行比較
            # 其差距最小的結果，即為我們要的結果
            # ex.
            # 傳送端送出的所有可能向量有 x1, x2, x3, x4   4種可能
            # 經過通道後的結果變成 y1, y2, y3, y4，而接收端實際收到的向量為y
            # 若 y 和 y2 的距離最近 (即 || y - y2 || 最小 )
            # 則我們會detect出傳送端是送x2向量


            # 定義一個ML detection
            # 利用遞迴函式來detect所有可能結果，並找出最佳解
            def ML_detection(H, detect, optimal_detection, y, current, min_distance, constellation):
            # H為通道矩陣、detect向量代表傳送端可能送出的向量、optimal_detection向量是存放detect後的傳送端最有可能送出的向量
            # y則是接收端實際收到的向量、current為目前遞迴到的位置、min_distance紀錄目前detection最小的距離差
            # constellation為星座點的集合
                Nt = H.shape[1]
                Nr = H.shape[0]
                if current == Nt:
                    # 找出detect向量和接收端收到的y向量間的距離
                    detect_y = [0] * Nr  # detect_y為detect向量經過通道矩陣後的結果
                    for i in range(Nr):
                        for j in range(Nt):
                            detect_y[i] += H[i, j] * detect[j]
                    # 接下來找出detect_y向量和y向量間距
                    s = 0
                    for i in range(Nr):
                        s += abs(y[i] - detect_y[i]) ** 2
                    s = np.sqrt(s)
                    # 所以s 即為兩向量間的距離的平方

                    # 如果detect出來的結果比之前的結果好，就更新optimal_detection向量
                    if s < min_distance[0]:
                        min_distance[0] = s
                        for i in range(Nt):
                            optimal_detection[i] = detect[i]
                else:
                    for i in range(len(constellation)):
                        detect[current] = constellation[i]
                        ML_detection(H, detect, optimal_detection, y, current + 1, min_distance, constellation)


            optimal_detection = [0]*Nt
            detect = [0]*Nt
            min_distance = [10**9]
            # 利用遞迴函式來detect所有可能結果，並找出最佳解
            ML_detection(H, detect, optimal_detection, y, 0, min_distance, constellation)

            # 接下來看錯多少個symbol
            if k==2 :#找BPSK錯幾個bit
                for m in range(Nt):
                    if optimal_detection[m] != symbol[m] :
                        error += 1
            elif k==3 :#找QPSK錯幾個bit
                for m in range(Nt):
                    if abs(optimal_detection[m].real - symbol[m].real) == 2:
                        error += 1
                    if abs(optimal_detection[m].imag - symbol[m].imag) == 2:
                        error += 1

        ber[i] = error / (K*Nt*N) #因為一個symbol有k個bit

    if k==0 :
        plt.semilogy(snr_db,ber,marker='o',label='theory (Nt=1 , Nr=1)')
    elif k==1 :
        plt.semilogy(snr_db, ber, marker='o', label='theory (Nt=1 , Nr=2  MRC)')
    elif k==2 :
        plt.semilogy(snr_db, ber, marker='o', label='ML detection for BPSK (Nt=2 , Nr=2 )')
    elif k==3 :
        plt.semilogy(snr_db, ber, marker='o', label='ML detection for QPSK (Nt=2 , Nr=2 )')

plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True,which='both')
plt.show()