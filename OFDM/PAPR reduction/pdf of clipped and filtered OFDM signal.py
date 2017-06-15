import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 該模擬的目的是看看將baseband OFDM signal乘上載波變成passband OFDM signal後
# 再對passband OFDN signal做clipping、filtering
# 觀察其pdf、PSD
# 還有觀察BPF (Band Pass Filter)的時域、頻域特性

Nfft = 128                  # 有Nfft個子載波
n_guard = Nfft//4           # guard interval
L = 8                       # oversampling factor
fc = 2e6                    # carrier frequency
Tsym = 128*(10**-6)         # OFDM symbol 周期
Ts = Tsym / Nfft            # 正常情況下的取樣間隔(即oversampling factor = 1)
Ts_oversample = Ts / L      # 在oversampling factor = L 情況下的取樣間隔
X = [0]*Nfft                # 存放Nfft個子載波的symbol
clipping_ratio = 1.2        # clipping level (限幅大小)與OFDM 取樣點的RMS(你可以想成是多個取樣點的標準差) 的比值



t = [0]*(Nfft + n_guard)*L  # 用來存放時間向量
# 每個過取樣點(oversampling points)的間隔為Ts_oversample
for i in range(len(t)):
    t[i] = i * Ts_oversample

normalize_t = [0]*(Nfft + n_guard)*L # 用來存放normalized後的時間向量
# 原本一個OFDM symbol在正常取樣下會有Nfft + n_guard個點
# 但經過oversampling後，會變成一個symbol有(Nfft + n_guard) * L 個點
for i in range(len(normalize_t)):
    normalize_t[i] = i / (Nfft*L)

f = [0]*(Nfft + n_guard)*L  # 用來存放頻率向量
# 因為每個過取樣點(oversampling points)間隔為Ts_oversample，所以我們可以找出其對應的frequency vector
# 其可分析的最大頻寬為 1 / Ts_oversample
j = -len(f) // 2
for i in range(len(f)):
    f[i] = j / (len(f) // 2) * (1/2*1/Ts_oversample) / (10 ** 6)
    j += 1
# 現在 f 向量的範圍在 ( - Ts_oversample / 2 ~ Ts_oversample / 2 )


constellation = [-1-1j, -1+1j, 1-1j, 1+1j]  # 決定星座點
K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量


# 我們先來設計濾波器吧
bpass = signal.remez(105, bands=[0/8, 1.4/8, 1.5/8, 2.5/8, 2.6/8, 4/8], desired=[0, 1, 0], weight=[10, 1, 10])
# 這個filter的order = 104，其產生的vector會有105個點，存放於bpass中
# bands代表濾波器中 0/8, 1.4/8 的一個band、1.5/8, 2.5/8 的一個band、2.6/8, 4/8 的一個band
# desired代表這三個band是否可以通過信號
# weight代表這三個band的weight
# 可參考  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.remez.html
H = np.fft.fft(bpass, (Nfft+n_guard)*L)

# 再來要對H排列一下，因為我們知道做FFT轉換後得到的頻域vector，其低頻處在兩端，高頻處在中間
# 我們現在要排列成高頻處在兩端，低頻處在中間
H = list(H[(len(H)+1)//2:]) + list(H[:(len(H)+1)//2])

# 接著找出H的絕對值平方並取其dB值
H_dB = [0]*len(H)
for i in range(len(H_dB)):
    H_dB[i] = 10*np.log10(abs(H[i])**2)

#決定所有sub-channel要送哪些信號
for m in range(Nfft):
    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
    for n in range(len(constellation)):
        if b <= (n + 1) / len(constellation):
            X[m] = constellation[n]
            break

# 接下來會用類似LPF的方法，在vector X中間補零，再做IFFT，達到時域siganl oversample的目的
oversampling_X = X[:Nfft//2] + [0]*(Nfft*L - Nfft) + X[Nfft//2:]

oversampling_base_x = np.fft.ifft(oversampling_X) * L / np.sqrt(Nfft / Es)   # 做IFFT頻域轉到時域
# 乘上 L / np.sqrt(Nfft / Es)  的目的是normalize一個OFDM symbol 的每一個取樣點
# 使得每一個取樣點的平均能量為1

# 接下來要加上cp
oversampling_base_x = list(oversampling_base_x[(Nfft - n_guard)*L:]) + list(oversampling_base_x[:])

# 接下來將baseband signal變換成passband signal
pass_x = [0]*len(oversampling_base_x)
for i in range(len(pass_x)):
    pass_x[i] = (oversampling_base_x[i] * np.exp(1j * 2*np.pi * fc * t[i]) ).real

# 接下來要對passband signal做clipping(限幅)
# clipping的意思就是限定signal的幅度為某一定值ex 正負A
# 當signal的振幅大於A時，就會定為A、signal的振幅小於-A時，就會定為-A
# A = clipping ratio * sigma
# 我們要從passband signal的取樣點來估計標準差sigma
sigma = 0
for i in range(len(pass_x)):
    sigma += abs(pass_x[i])**2
sigma /= len(pass_x)
sigma = np.sqrt(sigma)  # 估計出取樣點的標準差了
# 最後訂定clipping level : A
A = clipping_ratio * sigma

# 接下來對passband signal進行clipping
pass_clip_x = [0]*len(pass_x)
for i in range(len(pass_clip_x)):
    if pass_x[i] > A:
        pass_clip_x[i] = A
    elif pass_x[i] < -A:
        pass_clip_x[i] = -A
    else:
        pass_clip_x[i] = pass_x[i]

# 最後對clipped passband signal做filtering
pass_clip_filter_x = signal.lfilter(bpass, [1], pass_clip_x)
# 關於該函數的用法可參考官網 https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.lfilter.html




# 我們也可以找oversampling baseband signal 的PSD
# 先對oversampling_base_x做FFT得到oversampling_base_X
oversampling_base_X = np.fft.fft(oversampling_base_x)
# 再對其取絕對值平方，即可得classical PSD，並順便取其dB值
oversampling_base_PSD = [0]*len(oversampling_base_X)
oversampling_base_PSD_dB = [0]*len(oversampling_base_X)
for i in range(len(oversampling_base_PSD)):
    oversampling_base_PSD[i] = abs(oversampling_base_X[i])**2
# 還要把PSD normalize成最大值為1
for i in range(len(oversampling_base_PSD)):
    oversampling_base_PSD[i] /= max(oversampling_base_PSD)
# 最後可以找其dB值
for i in range(len(oversampling_base_PSD_dB)):
    oversampling_base_PSD_dB[i] = 10*np.log10(oversampling_base_PSD[i])
# 再來要對oversampling_base_PSD_dB排列一下，因為我們知道做FFT轉換後得到的頻域vector，其低頻處在兩端，高頻處在中間
# 我們現在要排列成高頻處在兩端，低頻處在中間
oversampling_base_PSD_dB = list(oversampling_base_PSD_dB[(len(oversampling_base_PSD_dB)+1)//2:]) + list(oversampling_base_PSD_dB[:(len(oversampling_base_PSD_dB)+1)//2])


# 我們也可以來找passband signal 的PSD
# 先對pass_x做FFT得到pass_X
pass_X = np.fft.fft(pass_x)
# 再對其取絕對值平方，即可得classical PSD，並順便取其dB值
pass_PSD = [0]*len(pass_X)
pass_PSD_dB = [0]*len(pass_X)
for i in range(len(pass_PSD)):
    pass_PSD[i] = abs(pass_X[i])**2
# 還要把PSD normalize成最大值為1
for i in range(len(pass_PSD)):
    pass_PSD[i] /= max(pass_PSD)
# 最後找其dB值
for i in range(len(pass_PSD_dB)):
    pass_PSD_dB[i] = 10*np.log10(pass_PSD[i])
# 再來要對pass_PSD_dB排列一下，因為我們知道做FFT轉換後得到的頻域vector，其低頻處在兩端，高頻處在中間
# 我們現在要排列成高頻處在兩端，低頻處在中間
pass_PSD_dB = list(pass_PSD_dB[(len(pass_PSD_dB)+1)//2:]) + list(pass_PSD_dB[:(len(pass_PSD_dB)+1)//2])


# 我們也可以來找clipped passband signal 的PSD
# 先對pass_clip_x做FFT得到pass_clip_X
pass_clip_X = np.fft.fft(pass_clip_x)
# 再對其取絕對值平方，即可得classical PSD，並順便取其dB值
pass_clip_PSD = [0]*len(pass_clip_X)
pass_clip_PSD_dB = [0]*len(pass_clip_X)
for i in range(len(pass_clip_PSD)):
    pass_clip_PSD[i] = abs(pass_clip_X[i])**2
# 還要把PSD normalize成最大值為1
for i in range(len(pass_clip_PSD)):
    pass_clip_PSD[i] /= max(pass_clip_PSD)
# 最後找其dB值
for i in range(len(pass_clip_PSD_dB)):
    pass_clip_PSD_dB[i] = 10*np.log10(pass_clip_PSD[i])
# 再來要對pass_clip_PSD_dB排列一下，因為我們知道做FFT轉換後得到的頻域vector，其低頻處在兩端，高頻處在中間
# 我們現在要排列成高頻處在兩端，低頻處在中間
pass_clip_PSD_dB = list(pass_clip_PSD_dB[(len(pass_clip_PSD_dB)+1)//2:]) + list(pass_clip_PSD_dB[:(len(pass_clip_PSD_dB)+1)//2])


# 我們也可以找clipped and filtered passband signal的PSD
# 先對pass_clip_filter_x做FFT得到pass_clip_filter_X
pass_clip_filter_X = np.fft.fft(pass_clip_filter_x)
# 再對其取絕對值平方，即可得classical PSD，並順便取其dB值
pass_clip_filter_PSD = [0]*len(pass_clip_filter_X)
pass_clip_filter_PSD_dB = [0]*len(pass_clip_filter_X)
for i in range(len(pass_clip_filter_PSD)):
    pass_clip_filter_PSD[i] = abs(pass_clip_filter_X[i])**2
# 還要把PSD normalize成最大值為1
for i in range(len(pass_clip_filter_PSD)):
    pass_clip_filter_PSD[i] /= max(pass_clip_filter_PSD)
# 最後找其dB值
for i in range(len(pass_clip_filter_PSD_dB)):
    pass_clip_filter_PSD_dB[i] = 10*np.log10(pass_clip_filter_PSD[i])
# 再來要對pass_clip_filter_PSD_dB排列一下，因為我們知道做FFT轉換後得到的頻域vector，其低頻處在兩端，高頻處在中間
# 我們現在要排列成高頻處在兩端，低頻處在中間
pass_clip_filter_PSD_dB = list(pass_clip_filter_PSD_dB[(len(pass_clip_filter_PSD_dB)+1)//2:]) + list(pass_clip_filter_PSD_dB[:(len(pass_clip_filter_PSD_dB)+1)//2])



plt.figure(1)
plt.subplot(1,2,1)
scale = [0]*len(bpass)
for i in range(len(bpass)):
    scale[i] = i
plt.stem(scale, bpass)
plt.title('Filter coefficient ')
plt.xlabel('tap')
plt.ylabel('Filter coefficient h[n]')

plt.subplot(1,2,2)
plt.plot(f, H_dB)
plt.title('PSD of Filter')
plt.xlabel(r'$frequency(Hz)\/\/x10^6$')
plt.ylabel('PSD(dB)')

plt.figure(2)
plt.subplot(2,2,1)
oversampling_base_abs_x = [0]*len(oversampling_base_x)
for i in range(len(oversampling_base_abs_x)):
    oversampling_base_abs_x[i] = abs(oversampling_base_x[i])
plt.plot(normalize_t, oversampling_base_abs_x)
plt.title(r"$baseband\/\/signal\/\/x^{'}[n]\/\/,\/\/with\/\/CP$")
plt.xlabel('time (normalized by symbol duration)')
plt.ylabel(r"$|x^{'}[n]|$")

plt.subplot(2,2,2)
plt.hist(pass_x, 100, normed=True)
plt.title(r'$pdf\/\/of\/\/unclipped\/\/passband\/\/signal\/\/x^p[n]$')
plt.xlabel(r'$passband\/\/signal\/\/x^p[n]$')
plt.ylabel('pdf')

plt.subplot(2,2,3)
plt.plot(f, oversampling_base_PSD_dB)
plt.title(r"$PSD\/\/of\/\/baseband\/\/signal\/\/x^{'}[n]$")
plt.xlabel(r'$frequency(Hz)\/\/x10^6$')
plt.ylabel('PSD(dB)')

plt.subplot(2,2,4)
plt.plot(f, pass_PSD_dB)
plt.title(r"$PSD\/\/of\/\/unclipped\/\/passband\/\/signal\/\/x^{p}[n]$")
plt.xlabel(r'$frequency(Hz)\/\/x10^6$')
plt.ylabel('PSD(dB)')

plt.figure(3)
plt.subplot(2,2,1)
plt.hist(pass_clip_x, 100, normed=True)
plt.title(r'$pdf\/\/of\/\/clipped\/\/passband\/\/signal\/\/x^p_c[n]\/,\/clipping\/\/ratio={0}$'.format(clipping_ratio))
plt.xlabel(r'$passband\/\/clipped\/\/signal\/\/x^p_c[n]$')
plt.ylabel('pdf')

plt.subplot(2,2,2)
plt.hist(pass_clip_filter_x, 100, normed=True)
plt.title(r'$pdf\/\/of\/\/passband\/\/signal\/\/after\/\/clipping\/\/and\/\/filtering\/\/\~{{x^p_c}}[n]\/,\/clipping\/\/ratio={0}$'.format(clipping_ratio))
plt.xlabel(r'$passband\/\/clipped\/\/and\/\/filtered\/\/signal\/\/\~{x^p_c}[n]$')
plt.ylabel('pdf')

plt.subplot(2,2,3)
plt.plot(f, pass_clip_PSD_dB)
plt.title(r'$PSD\/\/of\/\/clipped\/\/passband\/\/signal\/\/x^p_c[n]$')
plt.xlabel(r'$frequency(Hz)\/\/x10^6$')
plt.ylabel('PSD(dB)')

plt.subplot(2,2,4)
plt.plot(f, pass_clip_filter_PSD_dB)
plt.title(r'$PSD\/\/of\/\/clipped\/\/and\/\/filtered\/\/passband\/\/signal\/\/\~{x^p_c}[n]$')
plt.xlabel(r'$frequency(Hz)\/\/x10^6$')
plt.ylabel('PSD(dB)')

plt.show()








