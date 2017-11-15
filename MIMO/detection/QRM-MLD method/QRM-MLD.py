import numpy as np
import matplotlib.pyplot as plt

snr_db = [0] * 12
snr = [0] * 12
ber = [0] * 12
visited_node = [0]*12
Nt = 2  # 傳送端天線數
Nr = 2  # 接收端天線數
N = 1000  # 執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2 * i
    snr[i] = np.power(10, snr_db[i] / 10)

# 這裡採用 Nt x Nr 的MIMO系統，所以原本通道矩陣為 Nr x Nt
# 但在sphere decoding中，因為我們會將向量取實部、虛部來組合，所以通道矩陣也會跟著變成2Nr x 2Nt 矩陣
H = [[0j] * Nt for i in range(Nr)]
H = np.matrix(H)
H_new = [[0j] * (2*Nt) for i in range(2*Nr)]
H_new = np.matrix(H_new)
symbol = [0] * Nt      # 因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
symbol_new = [0]*2*Nt  # 除此之外我們還採用將傳送向量取實部、虛部重新組合後，所以向量元素變兩倍
y = [0] * Nr           # 接收端的向量
y_new = [0]*2*Nr       # 將接收端的向量，對其取實部、虛部重新組合後得到的新向量

# 定義星座點，QPSK symbol值域為{1+j , 1-j , -1+j , -1-j }
# 則實部、虛部值域皆為{ -1, 1 }
constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
constellation_new = [-1, 1]
constellation_name = 'QPSK'
# 定義星座點，16QAM symbol值域為{1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j }
# 則實部、虛部值域皆為{ -3, -1, 1, 3}
constellation = [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
constellation_new = [-3, -1, 1, 3]
constellation_name = '16QAM'

# 定義64QAM星座點
constellation_new = [-7 , -5, -3, -1, 1, 3, 5, 7]
constellation_name = '64QAM'
constellation = []
for i in range(len(constellation_new)):
    for j in range(len(constellation_new)):
        constellation += [constellation_new[i] + 1j*constellation_new[j]]



soft = 1

plt.figure('BER')
plt.figure('Average visited node')

for k in range(3):
    for i in range(len(snr_db)):
        error = 0
        total = 0
        visit = 0 #用來紀錄經過幾個node

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)      # 平均一個symbol有Es的能量
        Eb = Es / K                           # 平均一個bit有Eb能量
        # 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大
        No = Eb / snr[i]                      # 最後決定K

        for j in range(N):
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



            # 接下來定義QRM-MLD的方法，其概念與sphere decoding類似
            def QRM_MLD(R, detect, optimal_detection, z, N, current, accumulated_metric, min_metric, zf_vector, constellation, soft=1):
                # R為H進行QR分解後的R矩陣、detect向量代表傳送端可能送出的向量(會藉由遞迴來不斷改變)、z向量為轉置後的Q矩陣乘上接收端收到的向量後的向量
                # N為傳送向量長度、current為目前遞迴到的位置
                # accumulated_metric為目前遞迴所累積到的metric，min_metric為遞迴到終點後最小的metric
                # zf_vector代表用zero forcing找出的vector
                # constellation為星座圖，也就是向量中元素的值域
                # soft值代表最多有幾條branch

                visit = 0    # visit用來紀錄經過幾個node

                if current < 0:
                    if accumulated_metric < min_metric[0]:  # 如果該條路徑終點的metric總和是目前metric中的最小值，則很有可能是答案，存起來！
                        min_metric[0] = accumulated_metric
                        for i in range(N):
                            optimal_detection[i] = detect[i]

                else:
                    # 先決定search tree目前這一層有多少branch，branch有哪些
                    # soft值代表有幾條branch，branch數目至少兩條，最多不超過新的星座點的數目
                    if soft < 1 :
                        soft = 1
                    elif soft > len(constellation):
                        soft = len(constellation)

                    branch = [0]*soft
                    # 找出zf_vector的第current個元素與哪個新的星座點最近，與哪個新的星座點第二近，與哪個新的星座點第三近...
                    distance = [[0]*2 for i in range(len(constellation))]
                    for i in range(len(constellation)):
                        distance[i][1] = i                                              # 紀錄這個距離是屬於 第i個新的星座點與zf_vector向量第current個元素的距離
                        distance[i][0] = abs(constellation[i] - zf_vector[current,0])     # 第i個新的星座點與zf_vector向量第current個元素的距離
                    distance.sort(key=lambda cust:cust[0])                             # 將距離從小大到排列

                    # 知道哪些新的星座點與zf_vector的第current個元素較近
                    # 我們可以決定branch為何
                    for i in range(len(branch)):
                        branch[i] = constellation[distance[i][1]]
                    # 決定完branch值了
                    # 舉個例 若新的星座點為[ -3, -1, 1, 3]，在"搜尋樹"的這一層level中對應的zf_vector元素為1.3
                    # 當soft = 2時，星座點中的 -1 , 1與1.3距離最近
                    # 所以branch定為 [ -1, 1 ]

                    for i in range(len(branch)):
                        detect[current] = branch[i]
                        # 接下來計算一下，若接收向量該位置的元素為branch[i]，則累積的metrix為何
                        metric = z[current, 0]
                        for j in range(N - 1, current - 1, -1):
                            metric -= detect[j] * R[current, j]
                        metric = abs(metric) ** 2
                        metric += accumulated_metric
                        if metric < min_metric[0]:  # 只有在"目前累積metric" 小於 "最小metric"的情況下才能繼續遞迴往下搜尋
                            visit += 1
                            visit += QRM_MLD(R, detect, optimal_detection, z, N, current - 1, metric, min_metric, zf_vector, constellation, soft)
                return visit



            # 接下要先定義如何sphere decoding
            def sphere_decoding(R, detect, optimal_detection, z, N, current, accumulated_metric, min_metric, constellation):
                # R為H進行QR分解後的R矩陣、detect向量代表傳送端可能送出的向量(會藉由遞迴來不斷改變)、z向量為轉置後的Q矩陣乘上接收端收到的向量後的向量
                # N為傳送向量長度、current為目前遞迴到的位置
                # accumulated_metric為目前遞迴所累積到的metric，min_metric為遞迴到終點後最小的metric
                # constellation為星座圖，也就是向量中元素的值域

                visit = 0   # visit用來紀錄經過幾個node

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

                # 使用QLM-MLD method
                Q, R = np.linalg.qr(H_new)
                detect = [0] * (Nt * 2)
                optimal_detection = [0] * (Nt * 2)
                z = Q.transpose() * (np.matrix(y_new).transpose())

                # 我們可以先來決定球的半徑 metric 為何  (可以降低複雜度)
                # 我們會先用zero forcing解出s向量
                s = R.I * z
                #s = (H_new.getH() * H_new).I * H_new.getH() * (np.matrix(y_new).transpose()) # s向量亦可用此方成式解出

                # 若不採用zero forcing解出s向量，而是令s為0向量時
                #s = np.matrix([0]*2*Nt).transpose()

                min_metric = 0
                for m in range(len(s)):
                    min_metric += abs(z[m] - s[m]) ** 2
                min_metric = [min_metric]

                # min_metric = [10 ** 9]  #不決定球半徑的方法
                visit += QRM_MLD(R, detect, optimal_detection, z, 2*Nt, 2*Nt-1, 0, min_metric, s, constellation_new, soft)

                ''''
                # 以下為測試可用注解忽略
                # 直接對s做hard decision
                for m in range(s.shape[0]):
                    # 利用maximum likelihood來detection
                    min_distance = 10**9
                    for n in range(len(constellation_new)):
                        if min_distance > abs(constellation_new[n] - s[m,0]):
                            min_distance = abs(constellation_new[n] - s[m,0])
                            optimal_detection[m] = constellation_new[n]
                            '''


            elif k == 1:
                # 使用sphere decoding
                Q, R = np.linalg.qr(H_new)
                detect = [0] * (Nt*2)
                optimal_detection = [0] * (Nt*2)
                z = Q.transpose() * (np.matrix(y_new).transpose())

                # 我們可以先來決定球的半徑 metric 為何  (可以降低複雜度)
                s = R.I*z
                min_metric = 0
                for m in range(len(s)):
                    min_metric += abs(z[m]-s[m])**2
                min_metric = [min_metric]

                #min_metric = [10 ** 9]  #不決定球半徑的方法
                visit += sphere_decoding(R, detect, optimal_detection, z, 2*Nt, 2*Nt-1, 0, min_metric, constellation_new)
            elif k == 2:
                # 使用ML detection
                detect = [0] * (Nt*2)
                optimal_detection = [0] * (Nt*2)
                min_distance = [10 ** 9]
                visit += ML_detection(H_new, detect, optimal_detection, y_new, 0, min_distance, constellation_new)


            if constellation_name == 'QPSK':
                # 接下來計算QPSK錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m] - symbol_new[m]) == 2:
                        error += 1

            elif constellation_name == '16QAM':
                # 接下來計算16QAM錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m] - symbol_new[m]) == 2 or abs(optimal_detection[m] - symbol_new[m]) == 6:
                        error += 1
                    if abs(optimal_detection[m] - symbol_new[m]) == 4:
                        error += 2

            elif constellation_name == '64QAM':
                # 接下來計算64QAM錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m] - symbol_new[m]) == 2 or abs(optimal_detection[m] - symbol_new[m]) == 6 or abs(optimal_detection[m] - symbol_new[m]) == 14:
                        error += 1
                    if abs(optimal_detection[m] - symbol_new[m]) == 4 or abs(optimal_detection[m] - symbol_new[m]) == 8 or abs(optimal_detection[m] - symbol_new[m]) == 12:
                        error += 2
                    if abs(optimal_detection[m] - symbol_new[m]) == 10:
                        error += 3



        ber[i] = error / (Nt * K * N)  # 除以K是因為一個symbol有K個bit
        visited_node[i] = visit / N

    if k == 0:
        plt.figure('BER')
        plt.semilogy(snr_db, ber, marker='o', label='{0} (QRM-MLD)\nsoft={1}'.format(constellation_name,soft))
        plt.figure('Average visited node')
        plt.plot(snr_db, visited_node, marker='o', label='{0} (QRM-MLD)\nsoft={1}'.format(constellation_name,soft))
    elif k == 1:
        plt.figure('BER')
        plt.semilogy(snr_db, ber, marker='o', label='{0} (sphere decoding)'.format(constellation_name))
        plt.figure('Average visited node')
        plt.plot(snr_db, visited_node, marker='o', label='{0} (sphere decoding)'.format(constellation_name))
    elif k == 2:
        plt.figure('BER')
        plt.semilogy(snr_db, ber, marker='o', label='{0} (ML decoding)'.format(constellation_name))
        plt.figure('Average visited node')
        plt.plot(snr_db, visited_node, marker='o', label='{0} (ML decoding)'.format(constellation_name))


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