import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 這個模擬我們會觀察ofdm symbol的power spectrum density (PSD)
# 其中有兩種方法來估測離散時域序列的PSD
# 最簡單的方法是直接將時域離散序列做FFT後，再取其絕對值平方，但也較不準確  ---->  classical spectrum estimation
# 最棒的方法是將離散的時域序列取autocorrelation，再對其做FFT  ---->  estimated by autocorrelation
# 而有些參考資料最後還會對PSD做normalize，這裡我們也會跟著實做

Nfft = 64                       # 有64個sub channel
Nusc = 52                       # 只有用到52個sub channel，假設sub channel 0,1,2,3,4,5,58,59,60,61,62,63不使用
n_guard = 16                    # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
N = 10000                       #執行N次來找power spectrum
X = [0]*64                      # 從頻域送出64個symbol
ofdm_symbol = []                # 時域上的ofdm symbol序列
segment_len = 1024              # 設定PSD的解析度
power = [0]*segment_len         # power spectrum
T_symbol = 3.2*(10**-6)         # ofdm symbol 周期
Ts = T_symbol / Nfft            # 取樣間隔


# 決定有哪些頻率
freq = [0] * segment_len  # power spectrum對應到的頻率
j = -(segment_len // 2)
for i in range(len(freq)):
    freq[i] = j / (segment_len // 2) * 32 / T_symbol / (10 ** 6)
    j += 1

constellation = [ -1, 1]  # 星座點

for k in range(2):
    for i in range(len(power)):
        power[i] = 0

    ofdm_symbol = []
    for i in range(N):
        #決定所有sub-channel要送哪些信號
        for m in range(64):#sub-channel :  0,1,2,3,4,5,59,60,61,62,63不用來傳送symbol
            if m>=6 and m<=58:
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break
            else:
                X[m] = 0

        # 將頻域的Nfft個 symbol 做 ifft 轉到時域
        x = np.fft.ifft(X)

        if k == 0:
            #接下來要加上cyclic prefix
            x_new = [0]*(Nfft+n_guard)
            n = 0
            for m in range(Nfft-n_guard,Nfft,1):
                x_new[n] = x[m]
                n += 1
            for m in range(Nfft):
                x_new[n] = x[m]
                n += 1
            x = x_new  #現在x已經有加上cyclic prefix

        elif k == 1:
            # 接下來要zero padding
            x_new = [0] * (Nfft + n_guard)
            n = 0
            for m in range(Nfft - n_guard, Nfft, 1):
                x_new[n] = 0
                n += 1
            for m in range(Nfft):
                x_new[n] = x[m]
                n += 1
            x = x_new  # 現在x已經有加上cyclic prefix

        ofdm_symbol += x

        # 利用fft直接來統計PSD(準確性較差)  ---->  classical spectrum estimation
        new_X = np.fft.fft(x,segment_len)
        for j in range(len(new_X)):
            power[j] += abs(new_X[j])**2


    # 該函數的功用是將一離散的時域序列ofdm_symbol，利用統計特性來求PSD(準確性較佳)  ---->  estimated by autocorrelation
    # nperseg為segment長度，取越大則PSD解析度越高，但variance也會越大
    # return_onesided則是設定是否用double side的PSD(注意若時序序列為複數序列，則強制變為double side PSD)
    f, PSD = signal.welch(ofdm_symbol,nperseg=segment_len,return_onesided=False)

    # 接下來要將PSD用mean power來normalize
    # 所以要先找出mean power
    # 總共有Nfft=64個sub channel，其中只用到第6~57個sub channel
    # 所以要統計這52個sub channel的平均功率
    mean_power = 0
    for j in range(len(PSD)):
        if (j % (segment_len // Nfft) == 0) and (j // (segment_len // Nfft) >= 6 and j // (segment_len // Nfft) <=58):
            mean_power += PSD[j]
    mean_power /= Nusc

    # 利用mean power來normalize PSD
    for j in range(len(PSD)):
        PSD[j] /= mean_power
        PSD[j] = 10 * np.log10(PSD[j])  # 最後取其dB值


    # 剛剛是我們對較佳估計的PSD做normalize，現在是對較差估計的PSD做normalize
    # 所以要先找出mean power
    # 總共有Nfft=64個sub channel，其中只用到第6~57個sub channel
    # 所以要統計這52個sub channel的平均功率
    mean_power = 0
    for j in range(len(power)):
        if (j % (segment_len // Nfft) == 0) and (j // (segment_len // Nfft) >= 6 and j // (segment_len // Nfft) <= 58):
            mean_power += power[j]
    mean_power /= Nusc

    # 利用mean power來normalize PSD
    for j in range(len(PSD)):
        power[j] /= mean_power
        power[j] = 10 * np.log10(power[j])  # 最後取其dB值


    if k == 0:
        plt.figure('estimated by autocorrelation')
        # 決定有哪些頻率
        freq = [0] * len(PSD)  # power spectrum對應到的頻率
        j = -(len(PSD) // 2)
        for i in range(len(freq)):
            #freq[i] = j / (len(PSD) // 2) * 32 / T_symbol / (10 ** 6)
            freq[i] = j / (len(PSD) // 2) * (1/2*1/Ts) / (10 ** 6)
            j += 1
        plt.plot(freq,PSD,label='with CP')

        plt.figure('classical spectrum estimation')
        plt.plot(freq,power,label='with CP')
    elif k == 1:
        plt.figure('estimated by autocorrelation')
        plt.plot(freq, PSD, label='zero padding')

        plt.figure('classical spectrum estimation')
        plt.plot(freq, power, label='zero padding')

plt.figure('estimated by autocorrelation')
plt.legend()
plt.ylabel('power spectral desinty (dB)')
plt.xlabel('frequency, MHz')
plt.grid(True,which='both')
plt.ylim(-35,max(power)+2)

plt.figure('classical spectrum estimation')
plt.legend()
plt.ylabel('power spectral desinty (dB)')
plt.xlabel('frequency, MHz')
plt.grid(True,which='both')
plt.ylim(-35,max(power)+2)
plt.show()













