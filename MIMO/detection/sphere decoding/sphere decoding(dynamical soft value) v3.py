import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# 開始記時
tstart = time.time()

snr_db = [0] * 13
snr = [0] * len(snr_db)
ber = [0] * len(snr_db)
visited_node = [0] * len(snr_db)
add_computation = [0] * len(snr_db)   # 用來記錄平均加法次數
mult_computation = [0] * len(snr_db)  # 用來記錄平均乘法次數
Nt = 2  # 傳送端天線數
Nr = 2  # 接收端天線數
N = 1000000  # 執行N次來找錯誤率


# 這裡採用 Nt x Nr 的MIMO系統，所以原本通道矩陣為 Nr x Nt
# 但在sphere decoding中，因為我們會將向量取實部、虛部來組合，所以通道矩陣也會跟著變成2Nr x 2Nt 矩陣
H = [[0j] * Nt for i in range(Nr)]
H = np.matrix(H)
H_new = [[0j] * (2*Nt) for i in range(2*Nr)]
H_new = np.matrix(H_new)
symbol = np.matrix([0j] * Nt).transpose()      # 因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
symbol_new = np.matrix([0j]*2*Nt).transpose()  # 除此之外我們還採用將傳送向量取實部、虛部重新組合後，所以向量元素變兩倍
y = np.matrix([0j] * Nr).transpose()           # 接收端的向量
y_new = np.matrix([0j]*2*Nr).transpose()       # 將接收端的向量，對其取實部、虛部重新組合後得到的新向量


# 利用constellation_num決定要用哪種星座點
constellation_num = 2
if constellation_num == 1:
    # 定義星座點，QPSK symbol值域為{1+j , 1-j , -1+j , -1-j }
    # 則實部、虛部值域皆為{ -1, 1 }
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    constellation_new = [-1, 1]
    constellation_name = 'QPSK'
elif constellation_num == 2:
    # 定義星座點，16QAM symbol值域為{1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j }
    # 則實部、虛部值域皆為{ -3, -1, 1, 3}
    constellation = [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
    constellation_new = [-3, -1, 1, 3]
    constellation_name = '16QAM'
elif constellation_num == 3:
# 定義64QAM星座點
    constellation_new = [-7, -5, -3, -1, 1, 3, 5, 7]
    constellation_name = '64QAM'
    constellation = []
    for i in range(len(constellation_new)):
        for j in range(len(constellation_new)):
            constellation += [constellation_new[i] + 1j * constellation_new[j]]


soft = 8 # 選擇幾個soft 值 (沒屁用了)
soft_vector = [1,1,1,1]  # 會決定每一層要搜尋幾個節點


# 在terminal顯示目前是跑哪一種調變的模擬，而且跑幾個點
print('{0}模擬 , N={1} , soft_vector = {2}'.format(constellation_name, N, soft_vector))
# 定義way為路徑搜尋的方式
# 1代表DFS、2代表Best First Search、3代表BFS(Breadth-First-Search)其中K1為最多搜尋的節點數
way = 3
K1 = 8
if way == 1:
    way_name = 'DFS'
    print(way_name)
elif way == 2:
    way_name = 'Best First Search'
    print(way_name)
elif way == 3:
    way_name = 'BFS'
    print('BFS , K={0}'.format(K1))



# 根據不同的調變設定snr 間距
for i in range(len(snr)):
    if constellation_name == 'QPSK':
        snr_db[i] = 2 * i
    elif constellation_name == '16QAM':
        snr_db[i] = 2.5 * i
    elif constellation_name == '64QAM':
        snr_db[i] = 3 * i
    snr[i] = np.power(10, snr_db[i] / 10)


K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 平均一個symbol有Es的能量
Eb = Es / K                           # 平均一個bit有Eb能量
# 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大


for k in range(2):
    for i in range(len(snr_db)):
        error = 0 # 用來紀錄錯幾個symbol
        total = 0
        complexity = [0] * 3 # 分別記錄經過幾個node、做幾次加法運算、做幾次乘法運算

        No = Eb / snr[i]                      # 決定雜訊No

        for j in range(N):
            for m in range(len(symbol)):  # 決定傳送向量，要送哪些實數元素
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m,0] = constellation[n]
                        break
                symbol_new[m,0] = symbol[m,0].real
                symbol_new[m + Nt,0] = symbol[m,0].imag

            # 決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                    H_new[m, n] = H[m, n].real
                    H_new[m, n + Nt] = -H[m, n].imag
                    H_new[m + Nr, n] = H[m, n].imag
                    H_new[m + Nr, n + Nt] = H[m, n].real

            # 接下來決定接收端收到的向量y_new (共有2Nr 的元素)
            y_new = H_new * symbol_new
            for m in range(2*Nr):
                y_new[m,0] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()


            # 接下要先定義如何sphere decoding (DFS版本)
            def sphere_decoding_dfs(R, zf, K, soft_vector, detect, optimal_detection, N, current, accumulated_metric, min_metric, complexity, constellation):
                # R為H進行QR分解後的R矩陣、zf向量為zero forcing detection後尚未demapping的結果、K為選擇branch時要選擇離soft value最近的K個節點
                # soft_vector 有N個元素，用來決定不同tree level時要搜尋幾個節點
                # detect向量代表傳送端可能送出的向量(會藉由遞迴來不斷改變)
                # optimal_detection向量為最終detection後得到的結果
                # N為傳送向量長度、current為目前遞迴到的位置
                # accumulated_metric為目前遞迴所累積到的metric，min_metric為遞迴到終點後最小的metric
                # complexity有3個元素，分別記錄經過幾個node、做幾次加法運算、做幾次乘法運算
                # constellation為星座圖，也就是向量中元素的值域

                K = soft_vector[current]                   # 在這一層的tree level要搜尋幾個節點
                if current < 0:
                    if accumulated_metric < min_metric[0]:  # 如果該條路徑終點的metric總和是目前metric中的最小值，則很有可能是答案，存起來！
                        min_metric[0] = accumulated_metric
                        for i in range(N):
                            optimal_detection[i,0] = detect[i,0]
                else:
                    # 遞迴到這一層時對應到的soft value為zf[current, 0]
                    # 我們接下來要看哪些星座點離zf[current, 0]最近
                    select = [0]*len(constellation)   # 若select[i] == 0 代表不選擇星座點i，若select[i] == 1 代表會選擇星座點i
                    lt = [[0]*2 for i in range(len(constellation))]
                    for i in range(len(constellation)):
                        lt[i][0] = i                                     # 代表這是第i個星座點，待會排序後可以知道到底哪個星座點與soft value最近
                        lt[i][1] = abs(zf[current,0] - constellation[i]) # 代表此星座點與這個soft value間的距離
                    lt.sort(key = lambda cust:cust[1])    # 以星座點與 soft value間的距離為依據來排序，間距越小者排越前面
                    # 若lt[0][0] = 3 代表星座點3與soft value最近
                    # 若lt[1][0] = 1 代表星座點1與soft value第二近
                    # 若lt[2][0] = 0 代表星座點0與soft value第三近

                    # 注意到我們現在只要選擇K個離soft value最近的星座點
                    # 所以現在開始來選擇
                    count = 0
                    for i in range(len(constellation)):
                        select[lt[i][0]] = 1
                        count += 1
                        if count >= K:
                            break

                    for i in range(len(constellation)):
                        if select[i] == 0: # 若此星座點離soft value太遠而無法選取時，就跳過吧
                            continue

                        detect[current,0] = constellation[i]  # 若此星做點離soft value較近，我們就可以選取
                        complexity[0] += 1  # 代表拜訪node點數加1

                        # 接下來計算一下，若接收向量該位置的元素為constellation[i]，則累積的metrix為何
                        metric = z[current, 0]
                        for j in range(N - 1, current - 1, -1):
                            metric -= detect[j,0] * R[current, j]  # 注意到每次都有一個減法 and 乘法運算
                            # 做一次乘法運算時，我們會採用以下的方法，所以記錄成做一次加法
                            # 因為detect[j,0]的值域在64QAM時為 -7, -5, -3, -1, 1, 3, 5, 7
                            # 乘上1，相當於0次加法
                            # 乘上3，相當往左shift一個bit(x2)再做一次加法 = 一次加法
                            # 乘上5，相當往左shift兩個bit(x4)再做一次加法 = 一次加法
                            # 乘上7，相當往左shift三個bit(x8)再做一次減法 = 一次加法
                            if detect[j,0] != 1 and detect[j,0] != -1:
                                complexity[1] += 1  # 代表加法次數加1

                            complexity[1] += 1      # 因為最後一定會再做一次減法，所以加法次數加1

                        metric = abs(metric) ** 2
                        complexity[2] += 1          # 取絕對值再平方，記做一次乘法運算

                        metric += accumulated_metric
                        if current != N-1:           # 如果不是第一層的遞迴
                            complexity[1] += 1       # 將metric值累加，算一次加法運算

                        if metric < min_metric[0]:  # 只有在"目前累積metric" 小於 "最小metric"的情況下才能繼續遞迴往下搜尋
                            sphere_decoding_dfs(R, zf, K, soft_vector, detect, optimal_detection, N, current - 1, metric, min_metric, complexity, constellation)


            # 我們也定義一個ML detection
            def ML_detection(H, detect, optimal_detection, y, current, min_distance, complexity, constellation):
                # H為通道矩陣、detect向量代表傳送端可能送出的向量、optimal_detection向量是存放detect後的傳送端最有可能送出的向量
                # y則是接收端實際收到的向量、current為目前遞迴到的位置、min_distance紀錄目前detection最小的距離差
                # complexity有3個元素，分別記錄經過幾個node、做幾次加法運算、做幾次乘法運算
                # constellation為星座點的集合

                Nt = H.shape[1]
                Nr = H.shape[0]
                if current == Nt:
                    # 找出detect向量和接收端收到的y向量間的距離
                    detect_y = H * detect  # detect_y為detect向量經過通道矩陣後的結果

                    # 接下來找出detect_y向量和y向量間距
                    s = 0
                    for i in range(Nr):
                        s += abs(y[i,0] - detect_y[i,0]) ** 2
                    # 所以s 即為兩向量間的距離的平方

                    # 如果detect出來的結果比之前的結果好，就更新optimal_detection向量
                    if s < min_distance[0]:
                        min_distance[0] = s
                        for i in range(Nt):
                            optimal_detection[i,0] = detect[i,0]
                else:
                    for i in range(len(constellation)):
                        complexity[0] += 1  # 代表拜訪node點數加1
                        detect[current,0] = constellation[i]
                        # 我們就參考sphere decoding的方法來計算各種複雜度

                        ML_detection(H, detect, optimal_detection, y, current + 1, min_distance, complexity, constellation)

            def best_first_search(R, zf, K, soft_vector, optimal_detection, N, complexity, constellation): # best first serach 其實類似BFS search，只不過變為priority queue
                # R為H進行QR分解後的R矩陣
                # zf向量為zero forcing detection後尚未demapping的結果
                # K為選擇branch時要選擇離soft value最近的K個節點、soft_vector 有N個元素，用來決定不同tree level時要搜尋幾個節點
                # optimal_detection向量為最終detection後得到的結果
                # N為傳送向量長度
                # complexity有3個元素，分別記錄經過幾個node、做幾次加法運算、做幾次乘法運算
                # constellation為星座圖，也就是向量中元素的值域

                priority_queue = []

                K = soft_vector[N-1]    # 在tree最上層要搜尋soft_vector[N-1]個節點

                # tree的第一層時對應到的soft value為zf[N-1, 0]
                # 我們接下來要看哪些星座點離zf[N-1, 0]最近
                select = [0] * len(constellation)  # 若select[i] == 0 代表不選擇星座點i，若select[i] == 1 代表會選擇星座點i
                lt = [[0] * 2 for i in range(len(constellation))]
                for i in range(len(constellation)):
                    lt[i][0] = i  # 代表這是第i個星座點，待會排序後可以知道到底哪個星座點與soft value最近
                    lt[i][1] = abs(zf[N-1, 0] - constellation[i])  # 代表此星座點與這個soft value間的距離
                lt.sort(key=lambda cust: cust[1])  # 以星座點與 soft value間的距離為依據來排序，間距越小者排越前面
                # 若lt[0][0] = 3 代表星座點3與soft value最近
                # 若lt[1][0] = 1 代表星座點1與soft value第二近
                # 若lt[2][0] = 0 代表星座點0與soft value第三近

                # 注意到我們現在只要選擇K個離soft value最近的星座點
                # 所以現在開始來選擇
                count = 0
                for i in range(len(constellation)):
                    select[lt[i][0]] = 1
                    count += 1
                    if count >= K:
                        break

                # 再來將tree中第一層離soft value zf[N-1, 0]較近的星座點丟到priority queue中
                for i in range(len(constellation)):
                    if select[i] == 0:  # 若星座點i 離soft value zf[N-1, 0]較遠，就不選擇此星座點了
                        continue
                    vector = np.matrix([0j]*(N)).transpose()
                    vector[N-1,0] = constellation[i]
                    heapq.heappush(priority_queue,(abs(z[N-1,0] - R[N-1,N-1]*constellation[i])**2, vector, 1))
                    # 將(  (abs(z[N-1,0] - R[N-1,N-1]*constellation[i])**2, vector, 1)   ) 丟到priority queue中
                    # 其中(abs(z[N-1,0] - R[N-1,N-1]*constellation[i])**2值越小的話，優先權越大
                    # vector則存放選擇後的向量結果
                    # 1代表這個向量只有1個元素

                    complexity[0] += 1  # 經過node點數加1
                    # 做一次乘法運算時，我們會採用以下的方法，所以記錄成做一次加法
                    # 因為detect[j,0]的值域在64QAM時為 -7, -5, -3, -1, 1, 3, 5, 7
                    # 乘上1，相當於0次加法
                    # 乘上3，相當往左shift一個bit(x2)再做一次加法 = 一次加法
                    # 乘上5，相當往左shift兩個bit(x4)再做一次加法 = 一次加法
                    # 乘上7，相當往左shift三個bit(x8)再做一次減法 = 一次加法
                    if constellation[i] != 1 and constellation[i] != -1:
                        complexity[1] += 1   # 代表加法運算次數加1
                    complexity[1] += 1  # 因為有做一次減法，所以加法運算次數加1
                    complexity[2] += 1  # 因為最後有取絕對值平方，所以乘法運算次數加1


                # 接下來可以開始進行best first search了 (其過程類似BFS，只不過此處的queue變為priority queue)
                while True:
                    #先取出priority queue中優先權最高的元素
                    first_element = heapq.heappop(priority_queue)
                    # first_element[0] 存放上一層的accumulated_metric
                    # first_element[1] 存放上一層的vector
                    # first_element[2] 則代表上一層的vector有幾個元素

                    K = soft_vector[N - 1 - first_element[2]]   # 這一層要搜尋K個節點

                    if first_element[2] == N: # 若搜尋完畢
                        break

                    # 搜尋此節點的下層節點
                    # tree的這一層對應到的soft value為zf[N-1-first_element[2], 0]
                    # 我們接下來要看哪些星座點離zf[N-1-first_element[2], 0]最近
                    select = [0] * len(constellation)  # 若select[i] == 0 代表不選擇星座點i，若select[i] == 1 代表會選擇星座點i
                    lt = [[0] * 2 for i in range(len(constellation))]
                    for i in range(len(constellation)):
                        lt[i][0] = i  # 代表這是第i個星座點，待會排序後可以知道到底哪個星座點與soft value最近
                        lt[i][1] = abs(zf[N - 1 - first_element[2], 0] - constellation[i])  # 代表此星座點與這個soft value間的距離
                    lt.sort(key=lambda cust: cust[1])  # 以星座點與 soft value間的距離為依據來排序，間距越小者排越前面
                    # 若lt[0][0] = 3 代表星座點3與soft value最近
                    # 若lt[1][0] = 1 代表星座點1與soft value第二近
                    # 若lt[2][0] = 0 代表星座點0與soft value第三近

                    # 注意到我們現在只要選擇K個離soft value最近的星座點
                    # 所以現在開始來選擇
                    count = 0
                    for i in range(len(constellation)):
                        select[lt[i][0]] = 1
                        count += 1
                        if count >= K:
                            break

                    for i in range(len(constellation)):
                        if select[i] == 0:  # 若星座點i 離soft value zf[N - 1 - first_element[2], 0]較遠，就不選擇此星座點了
                            continue
                        vector = np.matrix(first_element[1])
                        vector[N - 1 - first_element[2], 0] = constellation[i]
                        complexity[0] += 1  # 經過node點數加1

                        # 接下來計算accumulated_metric
                        accumulated_metric = z[N - 1 - first_element[2], 0]
                        for j in range(N-1, N - 2 - first_element[2], -1):
                            accumulated_metric -= R[N-1-first_element[2] , j] * vector[j,0]  # 注意到每次都有一個減法 and 乘法運算
                            # 做一次乘法運算時，我們會採用以下的方法，所以記錄成做一次加法
                            # 因為detect[j,0]的值域在64QAM時為 -7, -5, -3, -1, 1, 3, 5, 7
                            # 乘上1，相當於0次加法
                            # 乘上3，相當往左shift一個bit(x2)再做一次加法 = 一次加法
                            # 乘上5，相當往左shift兩個bit(x4)再做一次加法 = 一次加法
                            # 乘上7，相當往左shift三個bit(x8)再做一次減法 = 一次加法
                            if vector[j,0] != 1 and vector[j,0] != -1:
                                complexity[1] += 1  # 代表加法運算次數加1
                            complexity[1] += 1      # 因為有做一次減法，所以加法運算次數加1

                        accumulated_metric = accumulated_metric ** 2
                        complexity[2] += 1          # 因為做一次平方運算=一次乘法運算

                        heapq.heappush(priority_queue,(first_element[0]+accumulated_metric, vector, first_element[2]+1))
                        # accumulated_matrix 變為first_element[0]+accumulated_metric
                        # 所以要做一次加法運算
                        complexity[1] += 1

                # first_element[1] 這個向量即為我們要的答案，將此向量的值複製到optimal_detection向量中
                for i in range(len(optimal_detection)):
                    optimal_detection[i,0] = first_element[1][i,0]


            def sphere_decoding_bfs(R, zf, K, soft_vector, optimal_detection, K1, N, complexity, constellation):
                # R為H進行QR分解後的R矩陣、zf向量為zero forcing detection後尚未demapping的結果、K為選擇branch時要選擇離soft value最近的K個節點
                # soft_vector 有N個元素，用來決定不同tree level時要搜尋幾個節點
                # optimal_detection向量為最終detection後得到的結果、K1為BFS搜尋中最多准許有幾個node出現
                # N為傳送向量長度
                # complexity有3個元素，分別記錄經過幾個node、做幾次加法運算、做幾次乘法運算
                # constellation為星座圖，也就是向量中元素的值域

                # 此處的bfs比較特別，這種bfs在搜尋一層樹時，每一條節點最多只會往下選擇其中離soft value較近的K個node
                # 除此之外下一層最多總共只含K1個節點，所以我用priority queue來決定要選擇哪K1個node
                queue = [[],[]]
                # 有兩個queue，a是要負責pop元素，b則push元素
                # 等到其中a  queue的元素pop完後
                # 換b queue pop元素，a  queue push元素

                current = 0 # 利用current來決定目前是哪個queue要pop元素

                K = soft_vector[N - 1]  # 在tree最上層要搜尋soft_vector[N-1]個節點

                # tree的第一層時對應到的soft value為zf[N-1, 0]
                # 我們接下來要看哪些星座點離zf[N-1, 0]最近
                select = [0] * len(constellation)  # 若select[i] == 0 代表不選擇星座點i，若select[i] == 1 代表會選擇星座點i
                lt = [[0] * 2 for i in range(len(constellation))]
                for i in range(len(constellation)):
                    lt[i][0] = i  # 代表這是第i個星座點，待會排序後可以知道到底哪個星座點與soft value最近
                    lt[i][1] = abs(zf[N - 1, 0] - constellation[i])  # 代表此星座點與這個soft value間的距離
                lt.sort(key=lambda cust: cust[1])  # 以星座點與 soft value間的距離為依據來排序，間距越小者排越前面
                # 若lt[0][0] = 3 代表星座點3與soft value最近
                # 若lt[1][0] = 1 代表星座點1與soft value第二近
                # 若lt[2][0] = 0 代表星座點0與soft value第三近

                # 注意到我們現在只要選擇K個離soft value最近的星座點
                # 所以現在開始來選擇
                count = 0
                for i in range(len(constellation)):
                    select[lt[i][0]] = 1
                    count += 1
                    if count >= K:
                        break

                # 一開始將tree中第一層離soft value zf[N-1, 0]較近的節點丟到queue[current]中
                for i in range(len(constellation)):
                    if select[i] == 0:  # 若此點離soft value太遠就不選取
                        continue
                    vector = np.matrix([0j] * (N)).transpose()
                    vector[N - 1, 0] = constellation[i]
                    heapq.heappush(queue[current], (abs(z[N - 1, 0] - R[N - 1, N - 1] * constellation[i]) ** 2, vector, 1))
                    # 將(  (abs(z[N-1,0] - R[N-1,N-1]*constellation[i])**2, vector, 1)   ) 丟到priority queue中
                    # 其中(abs(z[N-1,0] - R[N-1,N-1]*constellation[i])**2值越小的話，優先權越大
                    # vector則存放選擇後的向量結果
                    # 1代表這個向量只有1個元素

                    complexity[0] += 1  # 經過node點數加1
                    # 做一次乘法運算時，我們會採用以下的方法，所以記錄成做一次加法
                    # 因為detect[j,0]的值域在64QAM時為 -7, -5, -3, -1, 1, 3, 5, 7
                    # 乘上1，相當於0次加法
                    # 乘上3，相當往左shift一個bit(x2)再做一次加法 = 一次加法
                    # 乘上5，相當往左shift兩個bit(x4)再做一次加法 = 一次加法
                    # 乘上7，相當往左shift三個bit(x8)再做一次減法 = 一次加法
                    if constellation[i] != 1 and constellation[i] != -1:
                        complexity[1] += 1  # 代表加法運算次數加1
                    complexity[1] += 1  # 因為有做一次減法，所以加法運算次數加1
                    complexity[2] += 1  # 因為最後有取絕對值平方，所以乘法運算次數加1

                # 接下來要開始將queue[current]的元素pop到另一個queue[(current+1)%2]中
                while True:

                    count = 0  # count用來紀錄queue[current] 目前pop幾個元素
                    while True: # 將queue[current]的元素pop出來，並根據此pop出來的元素從tree的下層選出其他節點加到queue[(current+1)%2]中
                                # 注意到queue[current]最多只會pop K個node

                        if len(queue[current]) == 0:  # 若該queue的元素都pop出來了，就直接break
                            break
                        elif count >= K1:               # 若已經從queue[current]中pop K1個元素(代表從這一層的節點選取K1個節點了)
                            # 將queue[current]清空後在break
                            while True:
                                heapq.heappop(queue[current])
                                if len(queue[current]) == 0:
                                    break
                            break

                        # 先取出priority queue中優先權最高的元素
                        first_element = heapq.heappop(queue[current])
                        # first_element[0] 存放上一層的accumulated_metric
                        # first_element[1] 存放上一層的vector
                        # first_element[2] 則代表上一層的vector有幾個元素

                        count += 1  # 因為queue[current] pop出一個元素了

                        K = soft_vector[N - 1 - first_element[2]]  # 這一層要搜尋K個節點

                        if first_element[2] == N: # 代表BFS已搜尋到樹的最底端，搜尋結束
                            break

                        # 接下來搜尋此節點的下層節點
                        # 搜尋前我們要看，哪一個下層節點離下層soft value值較近，我們只選取離soft value值較近的K個節點
                        # 我們接下來要看哪些星座點離zf[N - 1 - first_element[2], 0]最近
                        select = [0] * len(constellation)  # 若select[i] == 0 代表不選擇星座點i，若select[i] == 1 代表會選擇星座點i
                        lt = [[0] * 2 for i in range(len(constellation))]
                        for i in range(len(constellation)):
                            lt[i][0] = i  # 代表這是第i個星座點，待會排序後可以知道到底哪個星座點與soft value最近
                            lt[i][1] = abs(zf[N - 1 - first_element[2], 0] - constellation[i])  # 代表此星座點與這個soft value間的距離
                        lt.sort(key=lambda cust: cust[1])  # 以星座點與 soft value間的距離為依據來排序，間距越小者排越前面
                        # 若lt[0][0] = 3 代表星座點3與soft value最近
                        # 若lt[1][0] = 1 代表星座點1與soft value第二近
                        # 若lt[2][0] = 0 代表星座點0與soft value第三近

                        # 注意到我們現在只要選擇K個離soft value最近的星座點
                        # 所以現在開始來選擇
                        count1 = 0
                        for i in range(len(constellation)):
                            select[lt[i][0]] = 1
                            count1 += 1
                            if count1 >= K:
                                break

                        for i in range(len(constellation)):
                            if select[i] == 0: # 若該星座點離此層的soft value值較遠就不選取
                                continue
                            vector = np.matrix(first_element[1])
                            vector[N - 1 - first_element[2], 0] = constellation[i]
                            complexity[0] += 1  # 經過node點數加1

                            # 接下來計算accumulated_metric
                            accumulated_metric = z[N - 1 - first_element[2], 0]
                            for j in range(N - 1, N - 2 - first_element[2], -1):
                                accumulated_metric -= R[N - 1 - first_element[2], j] * vector[j, 0]  # 注意到每次都有一個減法 and 乘法運算
                                # 做一次乘法運算時，我們會採用以下的方法，所以記錄成做一次加法
                                # 因為detect[j,0]的值域在64QAM時為 -7, -5, -3, -1, 1, 3, 5, 7
                                # 乘上1，相當於0次加法
                                # 乘上3，相當往左shift一個bit(x2)再做一次加法 = 一次加法
                                # 乘上5，相當往左shift兩個bit(x4)再做一次加法 = 一次加法
                                # 乘上7，相當往左shift三個bit(x8)再做一次減法 = 一次加法
                                if vector[j, 0] != 1 and vector[j, 0] != -1:
                                    complexity[1] += 1  # 代表加法運算次數加1
                                complexity[1] += 1  # 因為有做一次減法，所以加法運算次數加1

                            accumulated_metric = accumulated_metric ** 2
                            complexity[2] += 1  # 因為做一次平方運算=一次乘法運算

                            heapq.heappush(queue[(current+1)%2], (first_element[0] + accumulated_metric, vector, first_element[2] + 1)) # 將一個節點push到queue[(current+1)%2]中
                            # accumulated_matrix 變為first_element[0]+accumulated_metric
                            # 所以要做一次加法運算
                            complexity[1] += 1

                    current = (current+1)%2  # 因為待會負責pop的queue會變成要push、而剛剛負責push的queue則要負責pop

                    if first_element[2] == N:
                        break

                # first_element[1] 這個向量即為我們要的答案，將此向量的值複製到optimal_detection向量中
                for i in range(len(optimal_detection)):
                    optimal_detection[i, 0] = first_element[1][i, 0]



            if k == 0:
                # 使用sphere decoding
                Q, R = np.linalg.qr(H_new)
                detect = np.matrix([0j] * (Nt * 2)).transpose()
                optimal_detection = np.matrix([0j] * (Nt * 2)).transpose()
                z = Q.transpose() * y_new

                #我們可以先來決定球的半徑 metric 為何  (可以降低複雜度)
                s = R.I*z #若直接對s做demapping 來解調，其結果即為zero forcing detection
                min_metric = 0
                for m in range(len(s)):
                    min_metric += abs(z[m,0]-s[m,0])**2
                min_metric = [min_metric]  # min_metric即為利用zero forcing detection決定出來的球半徑
                min_metric = [10 ** 9]     # 若採用不決定球半徑的方法時

                # 以下提供3種最基本的sphere decoding detection
                if way == 1:
                    sphere_decoding_dfs(R, s, soft, soft_vector, detect, optimal_detection, 2*Nt, 2*Nt-1, 0, min_metric, complexity, constellation_new) # 花一點時間
                elif way == 2:
                    best_first_search(R, s, soft, soft_vector, optimal_detection, 2*Nt, complexity, constellation_new) # 花最少時間
                elif way == 3:
                    sphere_decoding_bfs(R, s, soft, soft_vector, optimal_detection, K1, 2*Nt, complexity, constellation_new)  # 花超多時間


            elif k == 1:
                # 使用ML detection
                # 我們已經有先前的模擬結果了，不要再花時間跑了
                None
                '''
                detect = np.matrix([0j] * (Nt*2)).transpose()
                optimal_detection = np.matrix([0j] * (Nt*2)).transpose()
                min_distance = [10 ** 9]
                ML_detection(H_new, detect, optimal_detection, y_new, 0, min_distance, complexity, constellation_new)
                '''


            if k == 1:  # 因為不用真的去做ML detection的模擬，所以也不用花時間找錯幾個bit
                continue

            if constellation_name == 'QPSK':  # 計算QPSK錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m,0] - symbol_new[m,0]) == 2:
                        error += 1
            elif constellation_name == '16QAM':  # 計算16QAM錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m,0] - symbol_new[m,0]) == 2 or abs(optimal_detection[m,0] - symbol_new[m,0]) == 6:
                        error += 1
                    if abs(optimal_detection[m,0] - symbol_new[m,0]) == 4:
                        error += 2
            elif constellation_name == '64QAM':#找64QAM錯幾個bit
                for m in range(len(symbol_new)):
                    if abs(optimal_detection[m,0] - symbol_new[m,0]) == 2 or abs(optimal_detection[m,0] - symbol_new[m,0]) == 6 or abs(optimal_detection[m,0] - symbol_new[m,0]) == 14:
                        error += 1
                    elif abs(optimal_detection[m,0] - symbol_new[m,0]) == 4 or abs(optimal_detection[m,0] - symbol_new[m,0]) == 8 or abs(optimal_detection[m,0] - symbol_new[m,0]) == 12:
                        error += 2
                    elif abs(optimal_detection[m,0] - symbol_new[m,0]) == 10:
                        error += 3




        ber[i] = error / (K * Nt * N)  # 除以K是因為一個symbol有K個bits
        visited_node[i] = complexity[0] / N
        add_computation[i] = complexity[1] / N
        mult_computation[i] = complexity[2] / N

    if k == 0:
        if way == 3:   # 若採用BFS搜尋
            plt.figure('BER({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
            plt.semilogy(snr_db, ber, marker='o', label='{0} (sphere decoding) soft_vector = {1}, K={2}'.format(constellation_name, soft_vector, K1))
            plt.figure('Average visited node({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
            plt.plot(snr_db, visited_node, marker='o', label='{0} (sphere decoding) soft_vector={1}, K={2}'.format(constellation_name, soft_vector, K1))
            plt.figure('Addition complexity({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
            plt.plot(snr_db, add_computation, marker='o', label='{0} (sphere decoding) soft_vector={1}, K={2}'.format(constellation_name, soft_vector, K1))
            plt.figure('Multiplication complexity({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
            plt.plot(snr_db, mult_computation, marker='o', label='{0} (sphere decoding) soft_vector={1}, K={2}'.format(constellation_name, soft_vector, K1))
        else:
            plt.figure('BER({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
            plt.semilogy(snr_db, ber, marker='o', label='{0} (sphere decoding) soft_vector = {1}'.format(constellation_name, soft_vector))
            plt.figure('Average visited node({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
            plt.plot(snr_db, visited_node, marker='o', label='{0} (sphere decoding) soft_vector={1}'.format(constellation_name, soft_vector))
            plt.figure('Addition complexity({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
            plt.plot(snr_db, add_computation, marker='o', label='{0} (sphere decoding) soft_vector={1}'.format(constellation_name, soft_vector))
            plt.figure('Multiplication complexity({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
            plt.plot(snr_db, mult_computation, marker='o', label='{0} (sphere decoding) soft_vector={1}'.format(constellation_name, soft_vector))

        if N >= 1000000:  # 在很多點模擬分析的情況下，錯誤率較正確，我們可以將數據存起來，之後就不用在花時間去模擬
            if way == 3:  # 若採用BFS搜尋
                with open('sphere decoding  for {0}, soft _vector= {1}, {2}, K={3} (Nt={4}, Nr={5}).dat'.format(constellation_name, soft_vector, way_name, K1, Nt, Nr), 'w') as f:
                    f.write('snr_db\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(snr_db[m]))
                    f.write('\nber\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(ber[m]))
                    f.write('\nAverage visited node\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(visited_node[m]))
                    f.write("\nAddition complexity\n")
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(add_computation[m]))
                    f.write("\nMultiplication complexity\n")
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(mult_computation[m]))
            else:
                with open('sphere decoding  for {0}, soft _vector= {1}, {2} (Nt={3}, Nr={4}).dat'.format(constellation_name, soft_vector, way_name, Nt, Nr),'w') as f:
                    f.write('snr_db\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(snr_db[m]))
                    f.write('\nber\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(ber[m]))
                    f.write('\nAverage visited node\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(visited_node[m]))
                    f.write("\nAddition complexity\n")
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(add_computation[m]))
                    f.write("\nMultiplication complexity\n")
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(mult_computation[m]))

    elif k == 1:
        # 我們先前就已完成ML detection的模擬，直接拿來用吧
        with open('../ML detection/data/ML detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt, Nr)) as f:
            # 以下的步驟都是讀取數據
            f.readline()                                   # 這一行讀取到字串 "snr_db"
            snr_db_string = f.readline()[:-2]              # 這一行讀取到的是各個 snr 組成的字串
            snr_db_list = snr_db_string.split(' ')          # 將各snr 組成的字串分開
            for m in range(len(snr_db_list)):
                snr_db_list[m] = float(snr_db_list[m])
            f.readline()                                   # 這一行讀取到字串 "ber"
            ber_string = f.readline()[:-1]                 # 這一行讀取到的是個個 ber 組成的字串
            ber_list = ber_string.split(' ')                # 將各個ber 組成的字串分開
            for m in range(len(ber_list)):
                ber_list[m] = float(ber_list[m])

        if way == 3:
            plt.figure('BER({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
        else:
            plt.figure('BER({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
        plt.semilogy(snr_db_list, ber_list, marker='o', label='{0} (ML decoding)'.format(constellation_name))
        #plt.semilogy(snr_db, ber, marker='o', label='{0} (ML decoding)'.format(constellation_name))
        # ML detection 的拜訪點數就不印出來了
        #plt.figure('Average visited node')
        #plt.plot(snr_db, visited_node, marker='o', label='{0} (ML decoding)'.format(constellation_name))


#結束時間，並統計程式執行時間 (可以利用跑少數個點的所需時間，來估計完整模擬的實際所需時間)
tend = time.time()
total_time = tend - tstart
total_time = int(total_time)
day = 0
hour = 0
min = 0
sec = 0
if(total_time > 24*60*60):
    day = total_time // (24*60*60)
    total_time %= (24*60*60)
if(total_time > 60*60):
    hour = total_time // (60*60)
    total_time %= (60*60)
if(total_time > 60):
    min = total_time // 60
    total_time %= 60
sec = float(total_time) + (tend - tstart) - int(tend - tstart)
print("spend {0} day, {1} hour, {2} min, {3:0.3f} sec".format(day,hour,min,sec))

if way == 3:
    plt.figure('BER({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
else:
    plt.figure('BER({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True, which='both')

if way == 3:
    plt.figure('Average visited node({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
else:
    plt.figure('Average visited node({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
plt.xlabel('Eb/No , dB')
plt.ylabel('Average visited node')
plt.legend()
plt.grid(True, which='both')

if way == 3:
    plt.figure('Addition complexity({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
else:
    plt.figure('Addition complexity({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
plt.xlabel('Eb/No , dB')
plt.ylabel('Average number of additions')
plt.legend()
plt.grid(True, which='both')

if way == 3:
    plt.figure('Multiplication complexity({0}), soft_vector={1}, {2}, K={3}'.format(constellation_name, soft_vector, way_name, K1))
else:
    plt.figure('Multiplication complexity({0}), soft_vector={1}, {2}'.format(constellation_name, soft_vector, way_name))
plt.xlabel('Eb/No , dB')
plt.ylabel('Average number of multiplications')
plt.legend()
plt.grid(True, which='both')

plt.show()