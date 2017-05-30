import numpy as np
import matplotlib.pyplot as plt
import cmath

Nfft = 32                                       # 總共有多少個sub channel
Nusc = 32                                       # 總共有多少sub channel 真正的被用來傳送symbol
n_guard = Nfft//4                               # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
CFO = [0.1, 0.3, 1.3]                           # 這是normalized frequency offset

constellation = [-1-1j,-1+1j,1-1j,1+1j]         # 決定星座點

# 決定時間軸的刻度
time = [0]*(Nfft+n_guard)
for i in range(len(time)):
    time[i] = i

# 決定所有sub-channel要送哪些信號
for m in range(Nfft):  # 假設所有sub-channel 都有送symbol
    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
    for n in range(len(constellation)):
        if b <= (n + 1) / len(constellation):
            X[m] = constellation[n]
            break

x = list(np.fft.ifft(X)) # 將頻域的子載波中的symbol轉成時域的OFDM symbol

# 對時域信號加上CP
x = x[Nfft-n_guard:] + x[:]

# 接下來考慮不同的CFO
for i in range(len(CFO)):
    x_CFO = x[:]
    for j in range(len(x_CFO)):
        x_CFO[j] *= np.exp(1j * 2 * np.pi * CFO[i] * j / Nfft)
    # x_CFO為受到CFO影響後的OFDM symbol

    # 接下來要觀察未受CFO影響，和受到CFO影響的信號，其phase(相位)為何，並順便紀錄相位差
    phase = [0]*len(x)
    phase_CFO = [0]*len(x_CFO)
    phase_diff = [0]*len(x)
    for j in range(len(phase)):
        phase[j] = cmath.phase(x[j])
        phase_CFO[j] = cmath.phase(x_CFO[j])
        phase_diff[j] = phase_CFO[j] - phase[j]
        if phase_diff[j] > np.pi:
            phase_diff[j] -= 2*np.pi
        elif phase_diff[j] < -np.pi:
            phase_diff[j] += 2*np.pi

    plt.subplot(3, 2, 2*i+1)
    plt.plot(time, phase, label=r'$\epsilon=0$')
    plt.plot(time, phase_CFO, label=r'$\epsilon={0}$'.format(CFO[i]))
    plt.legend(loc='upper right')
    plt.xlabel('time (n)')
    plt.ylabel('phase (rad)')
    plt.xlim(0, Nfft + n_guard - 1)
    plt.grid(True, which='both')

    plt.subplot(3, 2, 2*i+2)
    plt.plot(time, phase_diff)
    plt.xlabel('time (n)')
    plt.ylabel('phase difference (rad)')
    plt.xlim(0, Nfft + n_guard - 1)
    plt.grid(True, which='both')

plt.show()