import numpy as np
import matplotlib.pyplot as plt
import cmath,math

# 以下的模擬為 frequency-domain estimation techniques for STO

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
STO = [-6, -4, 1, 4]                            # 定義STO
CFO = [0, 0.5, 0, 0.5]                          # 定義CFO (這是normalized frequency offset )
snr_db = 30                                     # 決定snr(dB)
snr = np.power(10,snr_db/10)                    # 決定snr
common_delay = (Nfft + n_guard)*2               # common delay的用意是在找OFDM symbol的starting point時，忽略第一個ofdm symbol，直接找第二個ofdm symbol的starting point
                                                # 注意此時一個OFDM symbol有2*(Nfft+n_guard)個取樣點，前Nfft+n_guard個點為training symbol，後Nfft+n_guard個點才是data symbol
constellation = [-1, 1]                         # 決定星座點
K = int(np.log2(len(constellation)))            # 代表一個symbol含有K個bit


for k in range(len(STO)):
    prob_phase_rotation = [0] * (Nfft + n_guard)         # 將機率歸0


    # 我們先來決定training symbol為何
    Train = [0]*Nfft
    # 假設training symbol的所有子載波都送出1
    for m in range(Nfft):
        Train[m] = 1
    train = np.fft.ifft(Train)

    # 接下來要加上cyclic prefix
    t_new = [0] * (Nfft + n_guard)
    n = 0
    for m in range(Nfft - n_guard, Nfft, 1):
        t_new[n] = train[m]
        n += 1
    for m in range(Nfft):
        t_new[n] = train[m]
        n += 1
    training_symbol = t_new  # 現在training symbol已經有加上cyclic prefix

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
            # 在加入之前要先加上training symbol
            s += training_symbol
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
        Es = (energy / len(s)) * (Nfft + n_guard) / Nfft  # 平均一個取樣點有(energy / len(s))的能量
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
        # 定義一種利用頻域中，子載波間的phase rotation來估計STO的STO估計技術
        def STO_by_phase_rotation(s, Nfft, n_guard, common_delay):
            # common_delay的功用是從第二個OFDM symbol開始估計STO，而不是從第一個OFDM symbol估計STO

            # y為第二個OFDM symbol，有Nfft+n_guard個取樣點
            y = [0]*(Nfft+n_guard)
            for i in range(len(y)):
                y[i] = s[common_delay + i]

            # 接下來對y去除CP
            y_new = [0] * Nfft
            n = 0
            for m in range(n_guard, Nfft + n_guard, 1):
                y_new[n] = y[m]
                n += 1
            y = y_new  # 現在y已經去除OFDM的cyclic prefix

            Y = np.fft.fft(y) # 將時域的Nfft個取樣點轉到頻域

            # 接下來會用頻域的相位差(phase difference)來估計STO
            STO = 0
            for i in range(Nfft - 1):
                STO += Nfft / (2 * np.pi) * cmath.phase(Y[i + 1] * Y[i].conjugate())
            STO /= (Nfft - 1)  # 取mean

            # 將STO取最接近的整數
            if abs(STO - math.floor(STO)) < abs(STO - (math.floor(STO) + 1)):
                STO_estimate = math.floor(STO)
            else:
                STO_estimate = math.floor(STO) + 1

            return STO_estimate

        # 利用頻域中，子載波間的phase rotation來估計STO的STO估計技術
        # 從這個方法的模擬結果我們發現僅適用STO <= 0的情況下
        STO_phase_rotation = STO_by_phase_rotation(s_add_noise, Nfft, n_guard, common_delay)  # STO_phase_rotation即為用此方法估計出來的STO
        prob_phase_rotation[common_delay - STO_phase_rotation - ((Nfft+n_guard)*3)//2] += 1  # 統計STO為STO_phase_rotation的次數



    # 處理統計出來的機率
    for m in range(Nfft+n_guard):
        prob_phase_rotation[m] /= N
    # 尋找擁有最高機率的位置
    maximum = 0
    for m in range(Nfft+n_guard):
        if prob_phase_rotation[m] > maximum:
            maximum = prob_phase_rotation[m]
            position1 = common_delay - m - ((Nfft+n_guard)*3)//2

    # 決定STO的刻度
    # 刻度為 -(Nfft+n_guard)//2 ,  -(Nfft+n_guard)//2+1 , -(Nfft+n_guard)//2+2 .......共Nfft+n_guard個
    scale = [0] * (Nfft + n_guard)
    for m in range(len(scale)):
        scale[m] = common_delay - m - ((Nfft+n_guard)*3)//2

    # 接下來畫出各種estimation方法的STO機率分布圖
    plt.figure('Probability1')
    plt.subplot(2,2,k+1)
    plt.bar(scale, prob_phase_rotation, color='green', label='phase-rotation')
    plt.title('STO estimation, STO={0}, CFO={1}'.format(STO[k], CFO[k]))
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.xticks([-10, 0, 10,position1])
    plt.xlim(-32, 32)
    plt.ylim(0, 1)

plt.show()



