import numpy as np
import matplotlib.pyplot as plt

Ts = 1              # 正常情況的取樣周期，同時傳送端也會每個Ts送一個symbol
L = 8               # oversampling factor
Fs = 1/Ts * L       # 取樣頻率 (乘上L是因為過取樣(oversampled)的原因)
Fc = 1              # 載波頻率

for k in range(3):
    if k == 0:
        constellation = [-1+0j, 1+0j]
        constellation_name = 'BPSK'
    elif k == 1:
        constellation = [-1-1j, -1+1j, 1-1j, 1+1j]
        constellation_name = 'QPSK'
    elif k == 2:
        constellation = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j]
        constellation_name = '16QAM'

    N_symbol = len(constellation)  # 用來代表傳送端會送多少個symbol

    n_time = [0]*(L*N_symbol)      # 用來代表過取樣的時間點
    for m in range(len(n_time)):
        n_time[m] = m * Ts/L

    t_time = [0]*(L*N_symbol*50)   # 用來近似連續信號的時間
    for m in range(len(t_time)):
        t_time[m] = m * Ts/(L*50)

    # 先來決定baseband signal的discrete time sequence
    symbol_sequence = constellation[:]    # 假設傳送端送出所有星座圖中的每個symbol，而這個離散序列就是symbol_sequence

    s = [0]*(L*N_symbol)                  # s就是對baseband的連續時間信號過取樣後的結果
    for m in range(len(symbol_sequence)):
        for n in range(L):                # L為oversampling factor (就是會過取樣多少倍)
            s[m*L + n] = symbol_sequence[m]

    s_power = [0]*(L*N_symbol)            # 這是將s的每個取樣點取絕對值平方，代表每個取樣點的能量
    for m in range(len(s)):
        s_power[m] = abs(s[m])**2

    # 最後還要算一下s的PAPR
    # 先算average power
    # 並順便找出peak power
    avg_power = 0
    peak_power = 0
    for m in range(len(s_power)):
        avg_power += s_power[m]
        if s_power[m] > peak_power:
            peak_power = s_power[m]
    avg_power /= len(s_power)
    PAPR = peak_power / avg_power
    PAPR_dB = 10*np.log10(PAPR)

    s_real = [0]*len(s)                    # s_real 就是過取樣信號s的實部
    s_imag = [0]*len(s)                    # s_imag 就是過取樣信號s的虛部
    for m in range(len(s)):
        s_real[m] = s[m].real
        s_imag[m] = s[m].imag

    plt.figure(constellation_name)
    plt.subplot(3,2,1)
    markerline, stemlines, baseline = plt.stem(n_time, s_real, markerfmt=' ')
    plt.setp(baseline, 'color', 'k')  # 設定底線顏色為黑
    plt.setp(stemlines, 'color', 'k') # 設定脈衝顏色為黑
    plt.title('{0}, {1} symbols, Ts={2}s, Fs={3}Hz, L={4}'.format(constellation_name, N_symbol, Ts, Fs, L))
    plt.ylabel(r'$\~s_I[n]$')
    plt.subplot(3,2,3)
    markerline, stemlines, baseline = plt.stem(n_time, s_imag, markerfmt=' ')
    plt.setp(baseline, 'color', 'k')  # 設定底線顏色為黑
    plt.setp(stemlines, 'color', 'k')  # 設定脈衝顏色為黑
    plt.ylabel(r'$\~s_Q[n]$')
    plt.subplot(3,2,5)
    markerline, stemlines, baseline = plt.stem(n_time, s_power, markerfmt=' ')
    plt.setp(baseline, 'color', 'k')  # 設定底線顏色為黑
    plt.setp(stemlines, 'color', 'k')  # 設定脈衝顏色為黑
    plt.title('PAPR={0:.3F}dB'.format(PAPR_dB))
    plt.xlabel('time(s)\nbaseband signal')
    plt.ylabel(r'$|\~s_I[n]|^2+|\~s_Q[n]|^2$')

    # 接下來決定passband signal的continuous time signal及過取樣後的discrete time sequence
    # 先決定continuous time signal吧 (在模擬中仍是discrete time sequence，只是時間點較密集，所以看不出來是離散信號)
    continuous_s = [0]*len(t_time)
    p = 0
    for m in range(len(symbol_sequence)):
        for n in range(len(t_time) // len(symbol_sequence)):
            continuous_s[p] = ( symbol_sequence[m] * np.exp(1j * 2*np.pi * Fc * t_time[p]) ).real
            p += 1

    # 決定對continuous time signal過取樣後的discrete time sequence
    discrete_s = [0]*len(n_time)
    p = 0
    for m in range(len(symbol_sequence)):
        for n in range(len(n_time) // len(symbol_sequence)):
            discrete_s[p] = ( symbol_sequence[m] * np.exp(1j * 2*np.pi * Fc * n_time[p]) ).real
            p += 1

    # 接下來對continuous time signal和discrete time sequence的每一個點取平方來得到每一點的能量
    continuous_s_power = [0]*len(t_time)
    discrete_s_power = [0]*len(n_time)
    for m in range(len(continuous_s_power)):
        continuous_s_power[m] = abs(continuous_s[m])**2
    for m in range(len(discrete_s_power)):
        discrete_s_power[m] = abs(discrete_s[m])**2

    # 最後透過discrete time sequence找passband signal 的PAPR
    # 先算average power
    # 並順便找出peak power
    avg_power = 0
    peak_power = 0
    for m in range(len(discrete_s_power)):
        avg_power += discrete_s_power[m]
        if discrete_s_power[m] > peak_power:
            peak_power = discrete_s_power[m]
    avg_power /= len(discrete_s_power)
    PAPR = peak_power / avg_power
    PAPR_dB = 10 * np.log10(PAPR)

    plt.figure(constellation_name)
    plt.subplot(2, 2, 2)
    plt.plot(t_time, continuous_s, color='red', linestyle='--')
    markerline, stemlines, baseline = plt.stem(n_time, discrete_s, markerfmt=' ')
    plt.setp(baseline, 'color', 'k')  # 設定底線顏色為黑
    plt.setp(stemlines, 'color', 'k')  # 設定脈衝顏色為黑
    plt.title('{0}, {1} symbols, Ts={2}s, Fs={3}Hz, L={4}'.format(constellation_name, N_symbol, Ts, Fs, L))
    plt.ylabel(r'$s[n]$')
    plt.subplot(2,2,4)
    plt.plot(t_time, continuous_s_power, color='red', linestyle='--')
    markerline, stemlines, baseline = plt.stem(n_time, discrete_s_power, markerfmt=' ')
    plt.setp(baseline, 'color', 'k')  # 設定底線顏色為黑
    plt.setp(stemlines, 'color', 'k')  # 設定脈衝顏色為黑
    plt.title('PAPR={0:.3F}dB'.format(PAPR_dB))
    plt.xlabel('time (s)\npassband signal')
    plt.ylabel(r'$|s[n]|^2$')

plt.show()



