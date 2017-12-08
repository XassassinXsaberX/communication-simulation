import numpy as np
import matplotlib.pyplot as plt
import time

# 開始記時
tstart = time.time()

snr_db = [0]*13
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 10000000  #執行N次來找錯誤率
Nt = 2       #傳送端天線數
Nr = 2       #接收端天線數

#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = np.matrix([0j]*Nt).T   #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = np.matrix([0j]*Nr).T       #接收端的向量

# 定義BPSK星座點
#constellation = [1,-1]
#constellation_name = 'BPSK'

# 利用constellation_num決定要用哪種星座點
constellation_num = 1
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


normalize = 1  # 決定接收端是否要對雜訊normalize (若為0代表不normalize，若為1代表要normalize)

# 在terminal顯示目前是跑哪一種調變的模擬，而且跑幾個點
print('{0} ML detection模擬 , N={1} , Nr={2}, Nt={3}'.format(constellation_name, N, Nr, Nt))
if normalize == 0:
    print("non-normalize")
else:
    print("normalize")

# 根據不同的調變設定snr 間距
for i in range(len(snr)):
    if Nt == 2:
        if constellation_name == 'QPSK':
            snr_db[i] = 2 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 2.5 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 3 * i
        else:
            snr_db[i] = 2 * i
    elif Nt == 3:
        if constellation_name == 'QPSK':
            snr_db[i] = 1.7 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 2.2 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 2.6 * i
        else:
            snr_db[i] = 2 * i
    elif Nt == 4:
        if constellation_name == 'QPSK':
            snr_db[i] = 1.5 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 1.9 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 2.3 * i
        else:
            snr_db[i] = 2 * i
    else:
        snr_db[i] = 2 * i
    snr[i] = np.power(10, snr_db[i] / 10)


K = int(np.log2(len(constellation))) #代表一個symbol含有K個bit
#接下來要算平均一個symbol有多少能量
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m])**2
Es = energy / len(constellation)  #平均一個symbol有Es的能量
Eb = Es / K                       #平均一個bit有Eb能量
#因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大


for k in range(3):
    for i in range(len(snr)):
        error = 0
        total = 0
        if k == 0:  # SISO for BPSK theory
            ber[i] = 1/2 - 1/2*np.power(1+1/snr[i], -1/2)
            continue
        elif k ==1 : # MRC (1x2) for BPSK theroy
            ber[i] = 1/2 - 1/2*np.power(1+1/snr[i], -1/2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        for j in range(N):
            if normalize == 0:
                No = Eb / snr[i]       # 決定雜訊No
            else:
                No = Eb / snr[i] * Nr

            for m in range(Nt):  # 傳送端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n+1)/len(constellation):
                        symbol[m,0] = constellation[n]
                        break

            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            #接下來決定接收端收到的向量
            y = H * symbol
            for m in range(Nr):
                y[m,0] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

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
                    detect_y = H * detect# detect_y為detect向量經過通道矩陣後的結果

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
                        detect[current,0] = constellation[i]
                        ML_detection(H, detect, optimal_detection, y, current + 1, min_distance, constellation)

            optimal_detection = np.matrix([0j]*Nt).T
            detect = np.matrix([0j]*Nt).T
            min_distance = [10**9]
            # 利用遞迴函式來detect所有可能結果，並找出最佳解
            ML_detection(H, detect, optimal_detection, y, 0, min_distance, constellation)

            # 接下來看錯多少個symbol
            if constellation_name == 'BPSK' :#找BPSK錯幾個bit
                for m in range(Nt):
                    if optimal_detection[m,0] != symbol[m,0] :
                        error += 1
            elif constellation_name == 'QPSK' :#找QPSK錯幾個bit
                for m in range(Nt):
                    if abs(optimal_detection[m,0].real - symbol[m,0].real) == 2:
                        error += 1
                    if abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 2:
                        error += 1
            elif constellation_name == '16QAM':#找16QAM錯幾個bit
                for m in range(Nt):
                    if abs(optimal_detection[m,0].real - symbol[m,0].real) == 2 or abs(optimal_detection[m,0].real - symbol[m,0].real) == 6:
                        error += 1
                    elif abs(optimal_detection[m,0].real - symbol[m,0].real) == 4:
                        error += 2
                    if abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 2 or abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 6:
                        error += 1
                    elif abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 4:
                        error += 2
            elif constellation_name == '64QAM':#找64QAM錯幾個bit
                for m in range(Nt):
                    if abs(optimal_detection[m,0].real - symbol[m,0].real) == 2 or abs(optimal_detection[m,0].real - symbol[m,0].real) == 6 or abs(optimal_detection[m,0].real - symbol[m,0].real) == 14:
                        error += 1
                    elif abs(optimal_detection[m,0].real - symbol[m,0].real) == 4 or abs(optimal_detection[m,0].real - symbol[m,0].real) == 8 or abs(optimal_detection[m,0].real - symbol[m,0].real) == 12:
                        error += 2
                    elif abs(optimal_detection[m,0].real - symbol[m,0].real) == 10:
                        error += 3
                    if abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 2 or abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 6 or abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 14:
                        error += 1
                    elif abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 4 or abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 8 or abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 12:
                        error += 2
                    elif abs(optimal_detection[m,0].imag - symbol[m,0].imag) == 10:
                        error += 3


        ber[i] = error / (K*Nt*N) #因為一個symbol有K個bit

    if k==0 :
        plt.semilogy(snr_db,ber,marker='o',label='theory (Nt=1 , Nr=1) SISO for BPSK')
    elif k==1 :
        None
        #plt.semilogy(snr_db, ber, marker='o', label='theory (Nt=1 , Nr=2)  MRC for BPSK')
    elif k==2 :
        plt.semilogy(snr_db, ber, marker='o', label='ML detection for {0} (Nt={1} , Nr={2} )'.format(constellation_name,Nt,Nr))
        # 將錯誤率的數據存成檔案
        if N >= 1000000:   # 在很多點模擬分析的情況下，錯誤率較正確，我們可以將數據存起來，之後就不用在花時間去模擬
            if normalize == 0:
                with open('ML detection for {0} (non-normalize)(Nt={1}, Nr={2}).dat'.format(constellation_name,Nt,Nr),'w') as f:
                    f.write('snr_db\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(snr_db[m]))
                    f.write('\nber\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(ber[m]))
            else:
                with open('ML detection for {0} (normalize)(Nt={1}, Nr={2}).dat'.format(constellation_name,Nt,Nr),'w') as f:
                    f.write('snr_db\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(snr_db[m]))
                    f.write('\nber\n')
                    for m in range(len(snr_db)):
                        f.write("{0} ".format(ber[m]))
    '''  #以下部分可省略
    elif k==3 :
        plt.semilogy(snr_db, ber, marker='o', label='ML detection for {0} (Nt=2 , Nr=2 )'.format(constellation_name))
        print('snr_db for {0}'.format(constellation_name))
        print(snr_db)
        print('ber for {0}'.format(constellation_name))
        print(ber)
    '''

#結束時間，並統計程式執行時間 (可以利用跑少數個點的所需時間，來估計完整模擬的實際所需時間)
tend = time.time()
total_time = tend - tstart
total_time = int(total_time)
day = 0
hour = 0
minute = 0
sec = 0
if(total_time > 24*60*60):
    day = total_time // (24*60*60)
    total_time %= (24*60*60)
if(total_time > 60*60):
    hour = total_time // (60*60)
    total_time %= (60*60)
if(total_time > 60):
    minute = total_time // 60
    total_time %= 60
sec = float(total_time) + (tend - tstart) - int(tend - tstart)
print("spend {0} day, {1} hour, {2} min, {3:0.3f} sec".format(day,hour,minute,sec))

plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.legend()
plt.grid(True,which='both')
plt.show()

