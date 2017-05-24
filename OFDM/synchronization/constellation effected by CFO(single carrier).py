import numpy as np
import matplotlib.pyplot as plt

# 本模擬的目的是要看在single carrier情況下，發生carrier frequency offset時，星座點的變化
# 目前尚未加入ofdm system

N = 10                                         # 做N次迭代來統計星座點
normalized_freq_offset = [0, 0.01, 0.05, 0.3, 0.5, 1, 1.7, 2] # normalized後的frequency offset
freq_offset = [0]*len(normalized_freq_offset)   # 實際的frequency offset
Nfft = 64                                       # 一口氣會送Nfft個symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
s = [0]*Nfft                                    # 用來記錄傳送端送哪些symbol
constellation_real = []                         # 存放最後星座圖結果的實部
constellation_imag = []                         # 存放最後星座圖結果的虛部

constellation = [-1-1j,-1+1j,1-1j,1+1j] # 決定星座點
constellation_name = 'QPSK'

for k in range(len(normalized_freq_offset)):
    constellation_real = []  # 將星座圖的實部清空
    constellation_imag = []  # 將星座圖的虛部清空
    for i in range(N):
        for j in range(Nfft):
            b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
            for n in range(len(constellation)):
                if b <= (n + 1) / len(constellation):
                    s[j] = constellation[n]
                    break

        # 接下來考慮carrier frequency offset
        for m in range(Nfft):
            s[m] *= np.exp(1j * 2 * np.pi * normalized_freq_offset[k] * m / Nfft)
            # 亦可寫成
            #s[m] *= np.exp(1j * 2 * np.pi * freq_offset[k] * m * t_sample)

        # 假如不考慮通道及雜訊

        # 接下來統計星座點
        for m in range(len(s)):
            constellation_real += [s[m].real]
            constellation_imag += [s[m].imag]

    plt.subplot(2, 4, k + 1)
    plt.scatter(constellation_real, constellation_imag, s=7, marker='o',
                label=r'$normalized\/freq\/offset\/\epsilon = {0}$'.format(normalized_freq_offset[k]))
    plt.title('constellation for {0}\nin single-carrier system'.format(constellation_name))
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.legend(loc='upper right')
    plt.grid(True, which='both')
    plt.axis('equal')
    plt.axis([-2, 2, -2, 2])

plt.show()


