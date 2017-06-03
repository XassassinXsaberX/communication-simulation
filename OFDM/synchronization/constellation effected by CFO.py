import numpy as np
import matplotlib.pyplot as plt

Nfft = 64                                       # 總共有多少個sub channel
Nusc = 52                                       # 總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
n_guard = 16                                    # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
N = 100                                          # 做N次迭代來找error
L = 1                                           # 假設有L條multipath
h = [0]*L                                       # 存放multipath通道的 impulase response
H = [0]*N                                       # 為一個list其中有N個元素(因為總共會傳送N個ofdm symbol)，其中每一個元素會對應到一個序列類型的物件，這是代表傳送某個ofdm symbolo時的channel frequency response
max_delay_spread = L-1                          # 最大的 time delay spread為L-1個取樣點的時間，所以會有L條路徑，分別為delay 0,  delay 1,  delay 2 .......delay L-1 時間單位
normalized_freq_offset = [0, 0.001, 0.01, 0.05, 0.5, 1, 1.7, 2] # normalized後的frequency offset
freq_offset = [0]*len(normalized_freq_offset)   # 實際的frequency offset
for i in range(len(freq_offset)):
    freq_offset[i] = normalized_freq_offset[i] / T_symbol
constellation_real = []                         # 存放最後星座圖結果的實部
constellation_imag = []                         # 存放最後星座圖結果的虛部


constellation = [-1-1j,-1+1j,1-1j,1+1j] # 決定星座點
constellation_name = 'QPSK'

for k in range(len(normalized_freq_offset)):
    constellation_real = []                             # 將星座圖的實部清空
    constellation_imag = []                             # 將星座圖的虛部清空
    s = [0] * ( (Nfft + n_guard)*N + max_delay_spread)  # 代表ofdm的時域symbol序列
    number_of_symbol = 0                                # 用來紀錄目前到第幾個ofdm symbol了
    for i in range(N):                                  # 做N次迭代，相當於一口氣送N個ofdm symbol
        # 決定所有sub-channel要送哪些信號
        for m in range(Nfft):  # 假設sub-channel : 0,1,2,3,4,5及58,59,60,61,62,63不用來傳送symbol
            if m >= 6 and m <= 57:
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break
            else:
                X[m] = 0

        # 將頻域的Nfft個 symbol 做 ifft 轉到時域
        x = np.fft.ifft(X) * Nfft  # 乘上Nfft後得到的x序列，才是真正將時域信號取樣後的結果，你可以參考我在symbol timing中的模擬結果

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



        # 接下來要考慮multipath通道效應
        for m in range(L):
            h[m] = 1 / np.sqrt(2) / np.sqrt(L) * np.random.randn() + 1j / np.sqrt(2) / np.sqrt(L) * np.random.randn()  # 產生一個非時變通道
            # h[m] 除上 np.sqrt(L) 是為了要normalize multipath channel 的平均功率成1
            for n in range(Nfft+n_guard):
                s[(i * (Nfft + n_guard)) + n + m] += x[n]*h[m]
        H[i] = list(np.fft.fft(h,Nfft))  # 將impulse response h 做FFT，並將結果存到 H[i] 中

    # 接下來考慮CFO(carrier frequency offset)造成的影響，相當於乘上np.exp(1j * 2*np.pi *normalized_ frequency_offset [k]* m / Nfft)
    # 注意，一定要建立好完整的OFDM symbol序列後才能開始考慮CFO造成的影響
    for m in range(len(s)):
        s[m] *= np.exp(1j * 2 * np.pi * normalized_freq_offset[k] * m / Nfft)
        # 亦可寫成
        #s[m] *= np.exp(1j * 2 * np.pi * freq_offset[k] * m * t_sample)


    for i in range(N): #總共收到N個ofdm symbol
        y = [0]*(Nfft+n_guard)
        # 接下來要開始取樣
        # 準確的在時間點 0 , t_sample , 2*t_sample ...... (Nfft + n_guard - 1)*t_sample取樣
        STO = 0
        for m in range(Nfft+n_guard):
            y[m] = s[i*(Nfft+n_guard) + m]


        # 接下來要對取樣後的(Nfft + n_guard)個點去除cyclic prefix，變為Nfft個點
        y_new = [0] * Nfft
        n = 0
        for m in range(n_guard, Nfft + n_guard, 1):
            y_new[n] = y[m]
            n += 1
        y = y_new  # 現在y已經去除OFDM的cyclic prefix

        Y = np.fft.fft(y) / Nfft  # 除上Nfft，是為了要補償一開始做IFFT時所乘上的Nfft
        for m in range(Nfft):
            if m >= 6 and m <= 57:
                detection = Y[m] / H[i][m]
                constellation_real += [detection.real]
                constellation_imag += [detection.imag]


    plt.subplot(2,4,k+1)
    plt.scatter(constellation_real, constellation_imag, s=7, marker='o',label=r'$normalized\/freq\/offset\/\epsilon = {0}$'.format(normalized_freq_offset[k]))
    plt.title('constellation for {0} in OFDM symtem'.format(constellation_name))
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.legend(loc='upper right')
    plt.grid(True, which='both')
    plt.axis('equal')
    plt.axis([-2, 2, -2, 2])

plt.show()




