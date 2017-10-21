#BPSK  的SISO、MRC(2x1)、MRC(4x1) 錯誤率
#(2x1)代表傳送端有兩根天線，接收端有一根天線
import numpy as np
import matplotlib.pyplot as plt


snr_db = [0]*12
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
Nt = 4          # 傳送端天線數
Nt_select = 2   # 實際上會從Nt跟天線中選Nt_select跟天線來送data
Nr = 2          # 接收端天線數
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr_db)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

# 決定BPSK的星座點
constellation = [-1, 1]
constellation_name = 'BPSK'
# 定義QPSK星座點
constellation = [ -1-1j, -1+1j, 1-1j, 1+1j ]
constellation_name="QPSK"
# 定義16QAM星座點
constellation = [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
constellation_name='16QAM'
'''
# 接著定義64QAM星座點
constellation_new = [-7 , -5, -3, -1, 1, 3, 5, 7]
constellation_name = '64QAM'
constellation = []
for i in range(len(constellation_new)):
    for j in range(len(constellation_new)):
        constellation += [constellation_new[i] + 1j*constellation_new[j]]
'''


for k in range(6):
    for i in range(len(snr)):

        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        # 先將所有可能的星座點能量全部加起來
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)  # 平均一個symbol有Es的能量
        Eb = Es / K                       # 平均一個bit有Eb能量
        No = Eb / snr[i]                  # 最後決定No

        if k == 0:  # SISO(theory)
            ber[i] = 1/2-1/2*np.power(1+1/snr[i],-1/2)
            continue
        elif k == 1:  # MRC(1x2) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i]*ber[i]*(1+2*(1-ber[i]))
            continue
        else:
            if k == 2 or k == 3 :# 此時不採用antenna selection，所以傳送端只有Nt_select根天線而不是Nt根天線

                # 此時傳送端不採用antenna selection，而接收端採用zero forcing or MMSE detection
                # 因為我們的data資料夾中有2x2及4x4的MIMO with zero forcing  & MMSE detection模擬結果
                # 所以可拿出來使用
                if Nr == 2 or Nr == 4:
                    if k == 2:
                        try:
                            f = open('./data/ZF detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt_select,Nr))
                            success = 1  # 成功讀取數據
                        except:
                            success = 0  # 讀取數據失敗
                    elif k == 3:
                        try:
                            f = open('./data/MMSE detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt_select, Nr))
                            success = 1  # 成功讀取數據
                        except:
                            success = 0  # 讀取數據失敗
                    if success == 1:
                        # 以下的步驟都是讀取數據
                        f.readline()
                        snr_db_string = f.readline()[:-2]
                        snr_db_list = snr_db_string.split(' ')
                        for m in range(len(snr_db_list)):
                            snr_db_list[m] = float(snr_db_list[m])
                        f.readline()
                        ber_string = f.readline()[:-2]
                        ber_list = ber_string.split(' ')
                        for m in range(len(ber_list)):
                            ber_list[m] = float(ber_list[m])
                        # 接下來利用讀出來的數據畫出圖形
                        if k == 2:
                            plt.semilogy(snr_db_list, ber_list, marker='o', label='ZF, Nt={0}, Nr={1}, for {2}'.format(Nt_select, Nr, constellation_name))
                        elif k == 3:
                            plt.semilogy(snr_db_list, ber_list, marker='o', label='MMSE, Nt={0}, Nr={1}, for {2}'.format(Nt_select, Nr, constellation_name))
                        f.close()   # 關閉檔案
                        break

                # 若data資料夾中沒有此模擬的數據就要真的來模擬了
                # 這裡採用 Nt_select x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt_select
                H = [[0j] * Nt_select for i in range(Nr)]
                H = np.matrix(H)
                symbol = [0] * Nt_select  # 因為傳送端有Nt_select根天線，而且接收端不採用任何分集技術，所以會送Nt_select個不同symbol
                y = [0] * Nr              # 接收端的向量


            else:   # 此時有採用antenna selection，所以傳送端共有Nt根天線，會從其中Nt_select根天線來送data
                # 這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
                H = [[0j] * Nt for i in range(Nr)]                  # H代表尚未選擇天線前的channel matrix
                H = np.matrix(H)
                actual_H = [[0j] * Nt_select for i in range(Nr)]    # actual_H代表選擇天線後的channel matrix
                actual_H = np.matrix(actual_H)
                symbol = [0] * Nt_select  # 因為傳送端會選Nt_select根天線送data，而且接收端不採用任何分集技術，所以會送Nt_select個不同symbol
                y = [0] * Nr              # 接收端的向量

            for j in range(N):
                # 決定要送哪些symbol
                for m in range(Nt_select):  # 傳送端一次送出Nt_select個不同symbol
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                    for n in range(len(constellation)):
                        if b <= (n + 1) / len(constellation):
                            symbol[m] = constellation[n]
                            break

                # 接下來決定通道矩陣
                if k == 2 or k == 3:# 此時不採用antenna selection，所以傳送端只有Nt_select根天線而不是Nt根天線
                    for m in range(Nr):
                        for n in range(Nt_select):
                            H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

                else:               # 此時有採用antenna selection，所以傳送端共有Nt根天線，會從其中Nt_select根天線來送data
                    for m in range(Nr):
                        for n in range(Nt):
                            H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()
                    # 接下來選Nt_select個具有較大norm平方的column vector
                    # 也就是決定要送哪些天線
                    # ex . Nt = 4 , Nt_select = 2 , H = [ h0 , h1 , h2 , h3] 其中h1就是column vector
                    # 若 h0和h3的norm平方最大，則選擇天線0和天線3送data
                    # 所以實際上的channel matrix變為 [ h0 , h3 ]

                    norm_square = [[0]*2 for m in range(Nt)]  # 代表每一個column vector的norm平方，還有這是第幾個column vector
                    for m in range(Nt):
                        norm_square[m][1] = m  # 紀錄這是第幾個column vector
                        for n in range(Nr):
                            norm_square[m][0] += abs(H[n,m])**2  # 計算這個column vector的norm平方
                    # 接下來要對norm_square這個list進行排序
                    norm_square.sort(key = lambda cust:cust[0],reverse=True)

                    # 我們可以找actual_channel_matrix了
                    for m in range(Nt_select):
                        for n in range(Nr):
                            actual_H[n, m] = H[n, norm_square[m][1]]


                # 接下來決定接收端收到的向量y (共有Nr的元素)
                for m in range(Nr):
                    y[m] = 0
                for m in range(Nr):
                    for n in range(Nt_select):
                        if k == 2 or k == 3:  # 此時不採用antenna selection
                            y[m] += H[m, n] * symbol[n]
                        else:                 # 此時採用antenna selection
                            y[m] += actual_H[m, n] * symbol[n]
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

                if k == 2 or k == 3: # 此時不採用antenna selection
                    if k == 2:  # 執行ZF detection
                        # 決定ZF 的weight matrix
                        W = ((H.getH() * H) ** (-1)) * H.getH()  # W為 Nt _select x Nr 矩陣
                    elif k == 3:  # 執行MMSE detection
                        # 決定MMSE 的weight matrix
                        W = Es * (Es * H.getH() * H + No * np.identity(Nt_select)).I * H.getH()  # W為 Nt_select x Nr 矩陣
                else: # 此時採用antenna selection
                    if k == 4:  # 執行ZF detection
                        # 決定ZF 的weight matrix
                        W = ((actual_H.getH() * actual_H) ** (-1)) * actual_H.getH()  # W為 Nt_select x Nr 矩陣
                    elif k == 5:  # 執行MMSE detection
                        # 決定MMSE 的weight matrix
                        W = Es * (Es * actual_H.getH() * actual_H + No * np.identity(Nt_select)).I * actual_H.getH()  # W為 Nt_select x Nr 矩陣

                # receive向量 = W矩陣 * y向量
                receive = [0] * Nt_select
                for m in range(Nt_select):
                    for n in range(Nr):
                        receive[m] += W[m, n] * y[n]

                for m in range(Nt_select):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - receive[m]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - receive[m])
                    # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if symbol[m] != detection:
                        if constellation_name == 'BPSK':
                            error += 1  # error為bit error 次數
                        elif constellation_name == 'QPSK':
                            if abs(detection.real - symbol[m].real) == 2:
                                error += 1
                            if abs(detection.imag - symbol[m].imag) == 2:
                                error += 1
                        elif constellation_name == '16QAM':
                            if abs(detection.real - symbol[m].real) == 2 or abs(detection.real - symbol[m].real) == 6:
                                error += 1
                            elif abs(detection.real - symbol[m].real) == 4:
                                error += 2
                            if abs(detection.imag - symbol[m].imag) == 2 or abs(detection.imag - symbol[m].imag) == 6:
                                error += 1
                            elif abs(detection.imag - symbol[m].imag) == 4:
                                error += 2
                        elif constellation_name == '64QAM':
                            if abs(detection.real - symbol[m].real) == 2 or abs(detection.real - symbol[m].real) == 6 or abs(detection.real - symbol[m].real) == 14:
                                error += 1
                            elif abs(detection.real - symbol[m].real) == 4 or abs(detection.real - symbol[m].real) == 8 or abs(detection.real - symbol[m].real) == 12:
                                error += 2
                            elif abs(detection.real - symbol[m].real) == 10:
                                error += 3
                            if abs(detection.imag - symbol[m].imag) == 2 or abs(detection.imag - symbol[m].imag) == 6 or abs(detection.imag - symbol[m].imag) == 14:
                                error += 1
                            elif abs(detection.imag - symbol[m].imag) == 4 or abs(detection.imag - symbol[m].imag) == 8 or abs(detection.imag - symbol[m].imag) == 12:
                                error += 2
                            elif abs(detection.imag - symbol[m].imag) == 10:
                                error += 3

            ber[i] = error / (K * Nt_select * N)  # 除以K是因為一個symbol有K個bit




    if k == 0:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label="SISO for BPSK(theory)")
    elif k == 1:
        plt.semilogy(snr_db, ber, marker='o', linestyle='-', label='MRC(1x2) for BPSK(theory)')
    elif k == 2:
        if success == 0:  # 如果沒有data資料夾中沒有這個模擬，代表剛才是沒有畫圖，所以現在要畫圖
            plt.semilogy(snr_db, ber, marker='o', label='ZF, Nt={0}, Nr={1}, for {2}'.format(Nt_select, Nr, constellation_name))
    elif k == 3:
        if success == 0:  # 如果沒有data資料夾中沒有這個模擬，代表剛才是沒有畫圖，所以現在要畫圖
            plt.semilogy(snr_db, ber, marker='o', label='MMSE, Nt={0}, Nr={1}, for {2}'.format(Nt_select, Nr, constellation_name))
    elif k == 4:
        plt.semilogy(snr_db, ber, marker='o', label='ZF, Nt/Nt_select={0}/{1}, Nr={2}, for {3}\nwith antenna selection'.format(Nt, Nt_select, Nr, constellation_name))
    elif k == 5:
        plt.semilogy(snr_db, ber, marker='o', label='MMSE, Nt/Nt_select={0}/{1}, Nr={2}, for {3}\nwith antenna selection'.format(Nt, Nt_select, Nr, constellation_name))

plt.title('antenna selection for ZF/MMSE detection')
plt.ylabel("BER")
plt.xlabel("Eb/No , dB")
plt.grid(True,which='both')
plt.legend()
plt.show()