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


for k in range(3):
    for i in range(len(snr)):
        error = 0
        No = 1 / snr[i]  # 每個symbol只送一次能量
        # 已知 SNR = Eb / No
        # 令symbol 能量 Es =1 ，因為一次只送一個symbol所以Eb = Es = 1
        # 所以 No = 1 / SNR
        if k == 0:  # SISO theory
            ber[i] = 1/2 - 1/2*np.power(1+1/snr[i],-1/2)
            continue
        elif k ==1 : # MRC (1x2) theroy
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        for j in range(N):
            # 決定要送哪些symbol (採用BPSK調變)
            for m in range(Nt):  # 接收端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol[m] = 1
                else:
                    symbol[m] = -1

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


            # 利用遞迴函式來detect所有可能結果，並找出最佳解
            def find_all_possible(Nt,Nr,current,detect,H,min_distance,y,optimal_detection):
            # Nt 為傳送端天線數，Nr為接收端天線數，current為目前遞迴到的位置，detect向量代表傳送端可能送出的向量
            # H代表通道矩陣，min_distance紀錄目前detection最小的距離差，y則是接收端實際收到的向量
            # optimal_detection向量是存放detect後的傳送端最有可能送出的向量
                if current == Nt:
                    #找出detect向量和接收端收到的y向量間的距離
                    detect_y = [0]*Nr  #detect_y為detect向量經過通道矩陣後的結果
                    for i in range(Nr):
                        for j in range(Nt):
                            detect_y[i] += H[i,j]*detect[j]
                    #接下來找出detect_y向量和y向量間距
                    s = 0
                    for i in range(Nr):
                        s += abs(y[i]-detect_y[i])**2
                    s = np.sqrt(s)
                    #所以s 即為兩向量間的距離

                    #如果detect出來的結果比之前的結果好，就更新optimal_detection向量
                    if s < min_distance[0]:
                        min_distance[0] = s
                        for i in range(Nt):
                            optimal_detection[i] = detect[i]

                else:
                    for i in range(2):
                        if i == 0 :
                            detect[current] = 1
                            find_all_possible(Nt, Nr, current+1, detect, H, min_distance, y, optimal_detection)
                        else:
                            detect[current] = -1
                            find_all_possible(Nt, Nr, current+1, detect, H, min_distance, y, optimal_detection)


            optimal_detection = [0]*Nt
            detect = [0]*Nt
            min_distance = [10**9]
            # 利用遞迴函式來detect所有可能結果，並找出最佳解
            find_all_possible(Nt, Nr, 0, detect, H, min_distance, y, optimal_detection)

            # 接下來看錯多少個symbol
            for m in range(Nt):
                if optimal_detection[m] != symbol[m] :
                    error += 1

        ber[i] = error / (Nt*N)

    if k==0 :
        plt.semilogy(snr_db,ber,marker='o',label='theory (Nt=1 , Nr=1)')
    elif k==1 :
        plt.semilogy(snr_db, ber, marker='o', label='theory (Nt=1 , Nr=2  MRC)')
    elif k==2 :
        plt.semilogy(snr_db, ber, marker='o', label='ML detection (Nt=2 , Nr=2 )')

plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True,which='both')
plt.show()