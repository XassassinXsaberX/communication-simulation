import numpy as np
import matplotlib.pyplot as plt

snr_db = [0] * 12
snr = [0] * 12
ber = [0] * 12
visited_node = [0]*12
Nt = 2  # 傳送端天線數
Nr = 2  # 接收端天線數
N = 1000000  # 執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2 * i
    snr[i] = np.power(10, snr_db[i] / 10)

# 這裡採用 Nt x Nr 的MIMO系統，所以原本通道矩陣為 Nr x Nt
# 但在sphere decoding中，因為我們會將向量取實部、虛部來組合，所以通道矩陣也會跟著變成2Nr x 2Nt 矩陣
H = [[0j] * Nt for i in range(Nr)]
H = np.matrix(H)
H_new = [[0j] * (2*Nt) for i in range(2*Nr)]
H_new = np.matrix(H_new)
symbol = [0] * Nt  # 因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
symbol_new = [0]*2*Nt  # 除此之外我們還採用將傳送向量取實部、虛部重新組合後，所以向量元素變兩倍
y = [0] * Nr  # 接收端的向量，並對其取實部、虛部重新組合後得到的新向量
y_new = [0]*2*Nr

# 定義星座點，QPSK symbol值域為{1+j , 1-j , -1+j , -1-j }
# 則實部、虛部值域皆為{ -1, 1 }
constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
constellation_new = [-1, 1]

plt.figure('BER')
plt.figure('Average visited node')

for k in range(2):
    for i in range(len(snr_db)):
        error = 0
        total = 0
        visit = 0 #用來紀錄經過幾個node
        for j in range(N):
            # 已知 SNR = Eb / No
            # 令symbol 能量 Es =2 。採用QPSK調變，所以2個bit有2能量，平均1bit有1能量
            # 所以 No = 1 / SNR
            No = 1 / snr[i]  # 每個symbol只送一次能量

            for m in range(len(symbol)):  # 決定傳送向量，要送哪些實數元素
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break
                symbol_new[m] = symbol[m].real
                symbol_new[m + Nt] = symbol[m].imag

            # 決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                    H_new[m, n] = H[m, n].real
                    H_new[m, n + Nt] = -H[m, n].imag
                    H_new[m + Nr, n] = H[m, n].imag
                    H_new[m + Nr, n + Nt] = H[m, n].real

            # 接下來決定接收端收到的向量y (共有Nr 的元素)
            for m in range(Nr):
                y[m] = 0
            for m in range(Nr):
                for n in range(Nt):
                    y[m] += H[m, n]*symbol[n]
                y[m] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
                y_new[m] = y[m].real
                y_new[m+Nr] = y[m].imag



            # 接下要先定義如何sphere decoding
            def sphere_decoding(R, detect, optimal_detection, z, N, current, accumulated_metric, min_metric, constellation):
                # R為H進行QR分解後的R矩陣、detect向量代表傳送端可能送出的向量(會藉由遞迴來不斷改變)、z向量為轉置後的Q矩陣乘上接收端收到的向量後的向量
                # N為傳送向量長度、current為目前遞迴到的位置
                # accumulated_metric為目前遞迴所累積到的metric，min_metric為遞迴到終點後最小的metric
                # constellation為星座圖，也就是向量中元素的值域
                visit = 0
                # visit用來紀錄經過幾個node
                if current < 0:
                    if accumulated_metric < min_metric[0]:  # 如果該條路徑終點的metric總和是目前metric中的最小值，則很有可能是答案，存起來！
                        min_metric[0] = accumulated_metric
                        for i in range(N):
                            optimal_detection[i] = detect[i]
                else:
                    for i in range(len(constellation)):
                        detect[current] = constellation[i]
                        # 接下來計算一下，若接收向量該位置的元素為constellation[i]，則累積的metrix為何
                        metric = z[current, 0]
                        for j in range(N - 1, current - 1, -1):
                            metric -= detect[j] * R[current, j]
                        metric = abs(metric) ** 2
                        metric += accumulated_metric
                        if metric < min_metric[0]:  # 只有在"目前累積metric" 小於 "最小metric"的情況下才能繼續遞迴往下搜尋
                            visit += 1
                            visit += sphere_decoding(R, detect, optimal_detection, z, N, current - 1, metric, min_metric, constellation )
                return visit


            # 我們也定義一個ML detection
            def ML_detection(H, detect, optimal_detection, y, current, min_distance, constellation):
                # H為通道矩陣、detect向量代表傳送端可能送出的向量、optimal_detection向量是存放detect後的傳送端最有可能送出的向量
                # y則是接收端實際收到的向量、current為目前遞迴到的位置、min_distance紀錄目前detection最小的距離差
                # constellation為星座點的集合
                visit = 0
                # visit用來紀錄經過幾個node
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
                        visit += 1
                        visit += ML_detection(H, detect, optimal_detection, y, current + 1, min_distance, constellation)
                return visit


            if k == 0:
                # 使用sphere decoding
                Q, R = np.linalg.qr(H_new)
                detect = [0] * (Nt*2)
                optimal_detection = [0] * (Nt*2)
                z = Q.transpose() * (np.matrix(y_new).transpose())

                #我們可以先來決定球的半徑 metric 為何  (可以降低複雜度)
                s = R.I*z
                min_metric = 0
                for m in range(len(s)):
                    min_metric += abs(z[m]-s[m])**2
                min_metric = [min_metric]

                #min_metric = [10 ** 9]  #不決定球半徑的方法
                visit += sphere_decoding(R, detect, optimal_detection, z, 2*Nt, 2*Nt-1, 0, min_metric, constellation_new)
            elif k == 1:
                # 使用ML detection
                detect = [0] * (Nt*2)
                optimal_detection = [0] * (Nt*2)
                min_distance = [10 ** 9]
                visit += ML_detection(H_new, detect, optimal_detection, y_new, 0, min_distance, constellation_new)

            # 接下來計算QPSK錯幾個bit
            for m in range(len(symbol_new)):
                if abs(optimal_detection[m].real - symbol_new[m].real) == 2:
                    error += 1
                if abs(optimal_detection[m].imag - symbol_new[m].imag) == 2:
                    error += 1
            total += (2 * Nt)

        ber[i] = error / (2 * Nt * N)
        visited_node[i] = visit / N

    if k == 0:
        plt.figure('BER')
        plt.semilogy(snr_db, ber, marker='o', label='QPSK (sphere decoding)')
        plt.figure('Average visited node')
        plt.plot(snr_db, visited_node, marker='o', label='QPSK (sphere decoding)')
    elif k == 1:
        plt.figure('BER')
        plt.semilogy(snr_db, ber, marker='o', label='QPSK (ML decoding)')
        plt.figure('Average visited node')
        plt.plot(snr_db, visited_node, marker='o', label='QPSK (ML decoding)')

plt.figure('BER')
plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True, which='both')

plt.figure('Average visited node')
plt.xlabel('Eb/No , dB')
plt.ylabel('Average visited node')
plt.legend()
plt.grid(True, which='both')
plt.show()