import numpy as np
import matplotlib.pyplot as plt

# 以下的模擬為 time-domain estimation techniques for STO

Nfft = 128                                      # 總共有多少個sub channel
Nusc = 128                                      # 總共有多少sub channel 真正的被用來傳送symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
n_guard = Nfft//4                               # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
N = 100                                         # 做N次迭代 (當N=100時估計肯定正確)
symbol_number = 10                              # 一次送幾個OFDM symbol
L = 1                                           # 假設有L條multipath
h = [0]*L                                       # 存放multipath通道的 impulase response
H = [0]*N                                       # 為一個list其中有N個元素(因為總共會傳送N個ofdm symbol)，其中每一個元素會對應到一個序列類型的物件，這是代表傳送某個ofdm symbolo時的channel frequency response
max_delay_spread = L-1                          # 最大的 time delay spread為L-1個取樣點的時間，所以會有L條路徑，分別為delay 0,  delay 1,  delay 2 .......delay L-1 時間單位
STO = [-3, -3, 2, 2]                            # 定義STO
CFO = [0, 0.5, 0, 0.5]                          # 定義CFO (這是normalized frequency offset )
snr_db = 30                                     # 決定snr(dB)
snr = np.power(10,snr_db/10)                    # 決定snr
common_delay = (Nfft + n_guard)//2              # common delay的用意是在找OFDM symbol的starting point時，忽略第一個ofdm symbol
                                                # 直接找第二個ofdm symbol的starting point

constellation = [-1, 1]                         # 決定星座點
K = int(np.log2(len(constellation)))            # 代表一個symbol含有K個bit
prob_difference = [0]*(Nfft+n_guard)            # 用來統計利用兩個sample block的difference估計STO其結果為何的機率
prob_squared_difference = [0]*(Nfft+n_guard)    # 用來統計利用兩個sample block的squared difference估計STO其結果為何的機率
prob_correlation = [0]*(Nfft+n_guard)           # 用來統計利用兩個sample block的correlation估計STO其結果為何的機率


for k in range(len(STO)):
    average_difference = [0] * (Nfft + n_guard)          # 用來記錄兩個sample block的平均difference
    average_squared_difference = [0] * (Nfft + n_guard)  # 用來記錄兩個sample block的平均squared difference
    average_sum_of_correlation = [0] * (Nfft + n_guard)  # 用來記錄兩個sample block的平均auto correlation和(sum)
    prob_difference = [0] * (Nfft + n_guard)             # 將機率歸0
    prob_squared_difference = [0] * (Nfft + n_guard)     # 將機率歸0
    prob_correlation = [0] * (Nfft + n_guard)            # 將機率歸0

    for i in range(N):
        s = []  # 用來存放OFDM symbol

        # 接下來決定含有symbol_number個OFDM symbol的OFDM symbol序列
        for j in range(symbol_number):
            # 決定所有sub-channel要送哪些信號
            for m in range(Nfft):  # 假設所有sub-channel 都有送symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break

            # 將頻域的Nfft個 symbol 做 ifft 轉到時域
            x = np.fft.ifft(X) #* Nfft  # 乘上Nfft後得到的x序列，才是真正將時域信號取樣後的結果，你可以參考我在symbol timing中的模擬結果

            # 接下來要加上cyclic prefix
            x_new = [0] * (Nfft + n_guard)
            n = 0
            for m in range(Nfft - n_guard, Nfft, 1):
                x_new[n] = x[m]
                n += 1
            for m in range(Nfft):
                x_new[n] = x[m]
                n += 1
            x = x_new  # 現在x已經有加上cyclic prefix

            # 這裡先不考慮multipath通道吧

            # 將這一個OFDM symbol，加到完整的OFDM symbol 序列中
            s += x

        # 最後才考慮CFO (carrier frequency offset)造成的影響，相當於乘上np.exp(1j * 2*np.pi * CFO [k] * m / Nfft)
        # 其中CFO[k] 為normalized frequency offset
        # 注意，一定要加完cp後，並建立好完整的OFDM symbol序列後才能開始考慮CFO造成的影響
        for m in range(len(s)):
            s[m] *= np.exp(1j * 2 * np.pi * CFO[k] * m / Nfft)

        # 決定完完整的OFDM symbol 序列後
        # 接下來要統計平均一個symbol有多少能量
        energy = 0
        for m in range(len(s)):
            energy += abs(s[m]) ** 2
        Es = (energy / len(s)) * (Nfft+n_guard)/Nfft  # 平均一個取樣點有(energy / len(s))的能量
                                                      # 而平均一個symbol 的能量為平均一個取樣點能量乘上(Nfft + n_guard) / Nfft
        Eb = Es / K  # 平均一個bit 有Eb的能量
        No = Eb / snr

        # 接下來要加上 STO
        if STO[k] < 0:
            s = [0] * abs(STO[k]) + s[0:len(s) - abs(STO[k])]
        elif STO[k] > 0:
            s = s[abs(STO[k]):] + [0] * abs(STO[k])

        # 再來對這個OFDM symbol序列加上雜訊，並估計STO
        s_add_noise = [0]*len(s)
        # 加上Gaussian white noise
        for m in range(len(s_add_noise)):
            s_add_noise[m] = s[m] + np.sqrt(No/2) * np.random.randn() + 1j * np.sqrt(No/2) * np.random.randn()

        # 接下來要估計STO
        # 定義利用兩個sample block的difference來估計STO based on CP的函式
        def STO_by_difference(s, Nfft, n_guard, common_delay, difference):
            if common_delay == 0:
                common_delay = (Nfft + n_guard) / 2
            minimum = 10 ** 9
            for i in range(Nfft + n_guard):
                difference[i] = 0
                for j in range(n_guard):
                    # 下式是僅利用兩個sample block的difference來估計STO
                    difference[i] += abs(s[common_delay+i+j] - s[common_delay+i+j+Nfft])**2
                if difference[i] < minimum:
                    minimum = difference[i]
                    STO_estimate = common_delay - i
            return STO_estimate

        # 定義利用兩個sample block的squared difference來估計STO based on CP的函式
        def STO_by_square_difference(s, Nfft, n_guard, common_delay, squared_difference):
            if common_delay == 0:
                common_delay = (Nfft + n_guard) / 2
            minimum = 10 ** 9
            for i in range(Nfft + n_guard):
                squared_difference[i] = 0
                for j in range(n_guard):
                    # 下式是採用兩個sample block的squared difference來估計STO
                    squared_difference[i] += abs( abs(s[common_delay + i + j]) - abs(s[common_delay + i + j + Nfft])) ** 2
                if squared_difference[i] < minimum:
                    minimum = squared_difference[i]
                    STO_estimate = common_delay - i
            return STO_estimate

        # 定義利用correlation來估計STO based on CP的函式
        def STO_by_correlation(s, Nfft, n_guard, common_delay, sum_of_correlation):
            if common_delay == 0:
                common_delay = (Nfft + n_guard) / 2
            maximum = 0
            for i in range(Nfft + n_guard):
                sum_of_correlation[i] = 0
                for j in range(n_guard):
                    sum_of_correlation[i] += s[common_delay + i + j] * s[common_delay + i + j + Nfft].conjugate()
                    # s[common_delay+i+j] * s[common_delay+i+j+Nfft].conjugate() 代表s的auto correlation
                    # sum_of_correlation[i] 會將n_guard個不同的auto correlation相加
                sum_of_correlation[i] = abs(sum_of_correlation[i])
                if sum_of_correlation[i] > maximum:
                    maximum = sum_of_correlation[i]
                    STO_estimate = common_delay - i
            return STO_estimate

        # 利用兩個sample block的difference來估計STO
        # 這個方法的模擬結果不錯，可以正確估計STO來達到完全同步！
        difference = [0] * (Nfft + n_guard)
        STO_difference = STO_by_difference(s_add_noise, Nfft, n_guard, common_delay,difference)  # STO_difference即為用此方法估計出來的STO
        prob_difference[common_delay - STO_difference] += 1  # 統計STO為STO_difference的次數
        for m in range(len(difference)):
            average_difference[m] += difference[m]

        # 利用兩個sample block的squared difference來估計STO
        # 這個方法的模擬結果不錯，可以正確估計STO來達到完全同步！
        squared_difference = [0] * (Nfft + n_guard)
        STO_square_difference = STO_by_square_difference(s_add_noise, Nfft, n_guard, common_delay, squared_difference)  # STO_squared_difference即為用此方法估計出來的STO
        prob_squared_difference[common_delay - STO_square_difference] += 1  # 統計STO為STO_difference的次數
        for m in range(len(squared_difference)):
            average_squared_difference[m] += squared_difference[m]

        # 利用兩個sample block間的correlation來估計STO
        # 這是一個較差的估計方法，多做幾次模擬，你會發現每次估計出來的結果可能都不一樣(將迭代次數N設為1)....，但平均起來是正確的(將迭代次數N設為100)
        # 你也可以透過模擬出來的STO機率分布圖看出端倪
        sum_of_correlation = [0]*(Nfft+n_guard)
        STO_correlation = STO_by_correlation(s_add_noise, Nfft, n_guard, common_delay, sum_of_correlation) # STO_correlation即為用此方法估計出來的STO
        prob_correlation[common_delay - STO_correlation] += 1  # 統計STO為STO_correlation的次數
        for m in range(len(sum_of_correlation)):
            average_sum_of_correlation[m] += sum_of_correlation[m]


    for m in range(Nfft+n_guard):
        average_difference[m] /= N
        average_squared_difference[m] /= N
        average_sum_of_correlation[m] /= N

    # 找出average_sum_of_correlation最大的值及其對應的位置(position3)
    # 並找出average_squared_difference最小的值及其對應的位置(position2)
    maximum = 0
    minimum1 = 10**9
    minimum2 = 10**9
    for m in range(Nfft+n_guard):
        if average_difference[m] < minimum1:
            minimum1 = average_difference[m]
            position1 = common_delay - m
        if average_squared_difference[m] < minimum2:
            minimum2 = average_squared_difference[m]
            position2 = common_delay - m
        if average_sum_of_correlation[m] > maximum:
            maximum = average_sum_of_correlation[m]
            position3 = common_delay - m

    # 處理統計出來的機率
    for m in range(Nfft+n_guard):
        prob_difference[m] /= N
        prob_squared_difference[m] /= N
        prob_correlation[m] /= N

    # 決定STO的刻度
    # 刻度為 common_delay , common_delay-1 , common_delay-2 .......共Nfft+n_guard個
    scale = [0] * (Nfft + n_guard)
    for m in range(len(scale)):
        scale[m] = common_delay - m

    plt.figure('Magnitude')
    plt.subplot(2,2,k+1)
    plt.plot(scale, average_difference, color='green', label='min-difference')
    plt.plot(scale, average_squared_difference, color='red', label='min-squared-difference')
    plt.plot(scale, average_sum_of_correlation, color='blue', label='correlation-based')
    markerline, stemlines, baseline = plt.stem([position1], [minimum1])
    plt.setp(stemlines, color='green')  # 設定stem脈衝圖，脈衝的顏色為綠色
    markerline, stemlines, baseline = plt.stem([position2], [minimum2])
    plt.setp(stemlines, color='red')  # 設定stem脈衝圖，脈衝的顏色為紅色
    markerline, stemlines, baseline = plt.stem([position3], [maximum])
    plt.setp(stemlines, color='blue') # 設定stem脈衝圖，脈衝的顏色為藍色

    plt.title('STO estimation, STO={0}, CFO={1}'.format(STO[k], CFO[k]))
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right')
    plt.xticks([-10, 0, 10, position1, position2, position3])
    plt.xlim(-32,32)
    plt.ylim(0,max(max(average_difference),max(average_squared_difference),max(average_sum_of_correlation)))

    # 接下來畫出各種estimation方法的STO機率分布圖
    plt.figure('Probability1')
    plt.subplot(2,2,k+1)
    plt.bar(scale, prob_difference, color='green', label='min-difference')
    plt.title('STO estimation, STO={0}, CFO={1}'.format(STO[k], CFO[k]))
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.xticks([-10, 0, 10, position1])
    plt.xlim(-32, 32)
    plt.ylim(0, 1)

    plt.figure('Probability2')
    plt.subplot(2, 2, k + 1)
    plt.bar(scale, prob_squared_difference, color='red', label='min-squared-difference')
    plt.title('STO estimation, STO={0}, CFO={1}'.format(STO[k], CFO[k]))
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.xticks([-10, 0, 10, position2])
    plt.xlim(-32, 32)
    plt.ylim(0, 1)

    plt.figure('Probability3')
    plt.subplot(2, 2, k + 1)
    plt.bar(scale, prob_correlation, color='blue', label='correlation-based')
    plt.title('STO estimation, STO={0}, CFO={1}'.format(STO[k], CFO[k]))
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.xticks([-10, 0, 10, position3])
    plt.xlim(-32, 32)
    plt.ylim(0, 1)

plt.show()



