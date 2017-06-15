import numpy as np
import matplotlib.pyplot as plt

# 該模擬的目的是要觀察對baseband OFDM signal做clipping後，再量化
# 其量化誤差(即quantization noise)的程度
# 我們可發現clipping level太小時，SQNR越小(因為信號受到clipping distortion影響)
# 當clipping level太大時，SQNR也會越小(因為此時量化雜訊越大)

Nfft = 64                   # 有Nfft個子載波
X = [0]*Nfft                # Nfft個子載波要送出的symbol
L = 8                       # oversampling factor
bit = [6, 7, 8, 9]          # 用多少bit來量化時域的OFDM signal (補充一下，經取樣後的OFDM signal，每個取樣點都為高斯分佈)
u = [0]*40                  # 用來存放clipping level，clipping level指的是量化的最大值，所以當信號的振幅超過u時，經量化後變成不會超過u
normalized_u =[0]*len(u)    # clipping level normalized to sigma
sigma = np.sqrt(1/2)        # 將OFDM signal取樣後，每個取樣點的機率分佈為高斯分佈 N(0, sigma^2)
for i in range(len(u)):     # clipping level分別從 2 ~ 8
    normalized_u[i] = 2 + i*(8-2)/(len(u) - 1)
    u[i] = normalized_u[i] * sigma

SQNR = [0]*len(u)           # Signal-to-Quantization Noise Ratio
SQNR_dB = [0]*len(u)
iteration = 100000          # 迭代iteration次來找SQNR

constellation = [-1, 1]     # 定義星座點
K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量

for k in range(len(bit)):
    for i in range(len(u)):
        signal_power = 0
        quantization_nose_power = 0
        for j in range(iteration):
            # 決定所有sub-channel要送哪些信號
            for m in range(64):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break

            # 接下來會用類似LPF的方法，在vector X中間補零，再做IFFT，達到時域siganl oversample的目的
            oversampling_X = X[:Nfft//2] + [0]*(Nfft*L - Nfft) + X[Nfft//2:]

            oversampling_x = np.fft.ifft(oversampling_X) * (Nfft*L) / np.sqrt(Nfft/Es)  # 作IFFT頻域轉到時域
            # 乘上(Nfft*L) / np.sqrt(Nfft) / Es 的目的是將一個OFDM symbol的總能量normalize成 1* (Nfft*L)
            # 即平均每一個取樣點的平均能量為1

            #p = 0
            #for m in range(len(oversampling_x)):
            #    p += abs(oversampling_x[m].real) ** 2
            #p /= (Nfft*L - 1)


            # 接下來要將這些取樣點量化
            # 定義量化函數
            def quantization(oversampling_x, u, bit, quantization_x):
                # u代表clipping level、bit代表量化的位元數
                quantization_num = 2**bit               # 代表量化位階數，指的是可以分成幾個量化區域
                step_size = 2*u / (2**bit)              # 代表每個量化值的間隔
                quantization_rigion = [0]*(2**bit)      # 代表量化區域

                # 接下來決定每個量化區域的代表位階 (即將該區域的值統一視為某個數)
                quantization_rigion[0] = -u + step_size/2
                for i in range(1,2**bit,1):
                    quantization_rigion[i] = quantization_rigion[i-1] + step_size

                # 接下來可以開始量化
                for i in range(len(oversampling_x)):
                    if oversampling_x[i].real < quantization_rigion[0]:
                        quantization_x[i] = complex(quantization_rigion[0], quantization_x[i].imag)
                    elif oversampling_x[i].real > quantization_rigion[len(quantization_rigion)-1]:
                        quantization_x[i] = complex(quantization_rigion[len(quantization_rigion)-1], quantization_x[i].imag)

                    if oversampling_x[i].imag < quantization_rigion[0]:
                        quantization_x[i] = complex(quantization_x[i].real, quantization_rigion[0])
                        continue
                    elif oversampling_x[i].imag > quantization_rigion[len(quantization_rigion)-1]:
                        quantization_x[i] = complex(quantization_x[i].real, quantization_rigion[len(quantization_rigion)-1])
                        continue

                    for j in range(len(quantization_rigion)):
                        if abs(oversampling_x[i].real - quantization_rigion[j]) <= step_size / 2:
                            quantization_x[i] = complex(quantization_rigion[j], quantization_x[i].imag)
                            break
                    for j in range(len(quantization_rigion)):
                        if abs(oversampling_x[i].imag - quantization_rigion[j]) <= step_size / 2:
                            quantization_x[i] = complex(quantization_x[i].real, quantization_rigion[j])
                            break


            quantization_x = [0j]*len(oversampling_x)
            quantization(oversampling_x, u[i], bit[k], quantization_x)

            # 接下來要找signal power、quantization noise power
            for m in range(len(quantization_x)):
                signal_power += abs(oversampling_x[m])**2
                quantization_nose_power += abs(oversampling_x[m] - quantization_x[m])**2

        signal_power /= (iteration * L * Nfft)
        quantization_nose_power /= (iteration * L * Nfft)
        SQNR[i] = signal_power / quantization_nose_power
        SQNR_dB[i] = 10*np.log10(SQNR[i])


    plt.plot(normalized_u, SQNR_dB, marker='o', label='{0} bits quantization'.format(bit[k]))

plt.title(r'$Effect\/\/of\/\/Clipping\/\/(N_{{fft}}={0},\/\sigma^2=\frac{{1}}{{2}})$'.format(Nfft))
plt.xlabel(r'$u\/\/(clipping\/level\/normalized\/to\/\sigma)$')
plt.ylabel('SQNR(dB)')
plt.legend()
plt.grid(True,which='both')
plt.show()

