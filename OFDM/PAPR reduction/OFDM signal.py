import numpy as np
import matplotlib.pyplot as plt

# 此模擬的主要目的是要觀察傳送一連串OFDM symbol時
# 其時域信號經過過取樣(oversampled)後的取樣點其實部、虛部、振幅的機率密度
# 分別為高斯分佈、高斯分佈、Rayleigh 分佈

Nfft = 8                 # 子載波數目
Tc = 1                   # 一個OFDM symbol的週期
time= [0]*200            # 定義時間(從0秒到Tc秒)
for i in range(len(time)):
    time[i] = i*(Tc-0)/(len(time)-1)

constellation = [-1-1j, -1+1j, 1-1j, 1+1j]      # 定義星座點
constellation_name = 'QPSK'

X = [0]*Nfft                        # 有Nfft個子載波來送symbol
continuous_x = [0]*len(time)        # continuous_x代表baseband的continuous time OFDM signal
continuous_x_real = [0]*len(time)   # continuous_x_real代表baseband的continuous time OFDM signal的實部
continuous_x_imag = [0]*len(time)   # continuous_x_imag代表baseband的continuous time OFDM signal的虛部
continuous_x_abs = [0]*len(time)    # continuous_x_abs代表baseband的continuous time OFDM signal的振幅
continuous_single_carrier = [0]*Nfft           # 待會每個元素會變成一個list，分別代表不同的子載波乘上symbol後的時域信號
continuous_single_carrier_real = [0]*Nfft      # 待會每個元素會變成一個list，分別代表不同的子載波乘上symbol後的時域信號的實部
continuous_single_carrier_imag = [0]*Nfft      # 待會每個元素會變成一個list，分別代表不同的子載波乘上symbol後的時域信號的實部


#決定所有sub-channel要送哪些信號
for m in range(Nfft):
    if m != 0:            # 我們規定只有這幾個sub channel能傳送bpsk symbol
        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
        for n in range(len(constellation)):
            if b <= (n + 1) / len(constellation):
                X[m] = constellation[n]
                break
    else:
        X[m] = 0  # 直流的子載波不送symbol

# 接下來將X轉成時域的baseband continuous-time OFDM signal
# X[k]乘上子載波 1/Nfft * np.exp(1j * 2*np.pi * j/Tc * t) 其中Tc代表OFDM symbol period
for i in range(len(continuous_x)):
    for j in range(Nfft):
        continuous_x[i] += X[j] * 1/Nfft * np.exp(1j * 2*np.pi * j/Tc * time[i])

# 再來決定continuous_x_real、continuous_x_imag、continuous_x_abs
for i in range(len(continuous_x)):
    continuous_x_real[i] = continuous_x[i].real
    continuous_x_imag[i] = continuous_x[i].imag
    continuous_x_abs[i] = abs(continuous_x[i])

# 再來決定每個子載波乘上symbol所對應的時域信號，及其實部、虛部、振幅
#continuous_x = [0]*len(time)
for i in range(Nfft):
    signal = [0]*len(time)
    signal_real = [0]*len(time)
    signal_imag = [0]*len(time)
    for j in range(len(signal)):
        signal[j] = X[i] *  1/Nfft * np.exp(1j * 2*np.pi * i/Tc * time[j])
        signal_real[j] = signal[j].real
        signal_imag[j] = signal[j].imag
        #continuous_x[j] += signal[j]

    continuous_single_carrier[i] = signal
    continuous_single_carrier_real[i] = signal_real
    continuous_single_carrier_imag[i] = signal_imag

# 再來決定continuous_x_real、continuous_x_imag、continuous_x_abs
for i in range(len(continuous_x)):
    continuous_x_real[i] = continuous_x[i].real
    continuous_x_imag[i] = continuous_x[i].imag
    continuous_x_abs[i] = abs(continuous_x[i])

plt.subplot(3,2,1)
plt.plot(time, continuous_x_real)
for i in range(Nfft):
    plt.plot(time, continuous_single_carrier_real[i], color='black', linestyle='--', lw =0.5)
plt.title(r'${0}, N_{{FFT}}={1}$'.format(constellation_name, Nfft))
plt.ylabel(r'$x_I(t)$')

plt.subplot(3,2,3)
plt.plot(time, continuous_x_imag)
for i in range(Nfft):
    plt.plot(time, continuous_single_carrier_imag[i], color='black', linestyle='--', lw=0.5)
plt.ylabel(r'$x_Q(t)$')

plt.subplot(3,2,5)
plt.plot(time, continuous_x_abs)
plt.ylabel(r'$|x(t)|$')
plt.xlabel('time(s)\ntime-domain baseband OFDM signal')


# 再來要統計對連續信號取樣時，其實部、虛部、振幅的機率分佈
Nfft = 16      # 現在變成有16個子載波
N = 1000       # 做N次來找機率分佈
# Tc、time則沿用前面
X = [0]*Nfft                        # 有Nfft個子載波來送symbol

continuous_x = []                   # continuous_x代表N個baseband的continuous time OFDM signal

for k in range(N):
    # 決定所有sub-channel要送哪些信號
    for m in range(Nfft):
        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
        for n in range(len(constellation)):
            if b <= (n + 1) / len(constellation):
                X[m] = constellation[n]
                break

    # 接下來將X轉成時域的baseband continuous-time OFDM signal x
    # X[k]乘上子載波 1/Nfft * np.exp(1j * 2*np.pi * j/Tc * t) 其中Tc代表OFDM symbol period
    x = [0] * len(time)  # continuous-time OFDM signal
    for i in range(len(x)):
        for j in range(Nfft):
            x[i] += X[j] * 1/Nfft * np.exp(1j * 2*np.pi * j/Tc * time[i])

    continuous_x += x # 將x 為加到一連串的OFDM signal (continuous_x)中

continuous_x_real = [0]*len(continuous_x)   # continuous_x_real代表baseband的continuous time OFDM signal的實部
continuous_x_imag = [0]*len(continuous_x)   # continuous_x_imag代表baseband的continuous time OFDM signal的虛部
continuous_x_abs = [0]*len(continuous_x)    # continuous_x_abs代表baseband的continuous time OFDM signal的振幅
# 再來決定continuous_x_real、continuous_x_imag、continuous_x_abs
for i in range(len(continuous_x)):
    continuous_x_real[i] = continuous_x[i].real
    continuous_x_imag[i] = continuous_x[i].imag
    continuous_x_abs[i] = abs(continuous_x[i])

plt.subplot(3,2,2)
plt.hist(continuous_x_real, 50, normed=True)
plt.title(r'${0}, N_{{FFT}}={1}$'.format(constellation_name, Nfft))
plt.ylabel(r'$pdf\/\/of\/\/x_I(t)$')
plt.subplot(3,2,4)
plt.hist(continuous_x_imag, 50, normed=True)
plt.ylabel(r'$pdf\/\/of\/\/x_Q(t)$')
plt.subplot(3,2,6)
plt.hist(continuous_x_abs, 50, normed=True)
plt.ylabel(r'$pdf\/\/of\/\/|x(t)|$')
plt.xlabel('x\nMagnitude distribution of OFDM signal')




plt.show()