import numpy as np
import matplotlib.pyplot as plt
import math

Nfft = 16                 # 子載波數目
L = [1,4]                 # oversampling factor 分別為1和4，當oversampling factor > 1時，代表過取樣(oversample)
q = 3                     # Chu sequence的係數，q和Nfft的最大公因數需為1
time = [0]*len(L)         # 為一個list，list中的每個元素代表一個過取樣信號的時間序列(其為symbol duration normalized 成1後的時間序列)
X = [0]*Nfft              # 有Nfft個子載波來重送symbol

# 因為L有兩個元素，所以代表有兩種取樣結果
# 所以有兩種normalized time sequence
for i in range(len(L)):
    # 因為取樣結果會有Nfft * L[i]個點
    scale = [0]*(Nfft*L[i])
    for j in range(Nfft*L[i]):
        scale[j] = j / (Nfft*L[i] - 1)
    time[i] = scale

# 接下來決定Chu sequence
for i in range(Nfft):
    if Nfft % 2 == 0:
        X[i] = np.exp(-1j * np.pi * q * i*i / Nfft)
    else:
        X[i] = np.exp(-1j * np.pi * q * i*(i+1) / Nfft)

# 決定完Chu sequence後我們來看看不同的oversampling factor，其時域信號的差異
for i in range(len(L)):
    # 我們知道頻域的離散信號中(也就是vector)
    # 高頻處是位在中間的位置，而兩端為低頻處
    # ex.  X = [1, 2, 0, 0, 0, 0, 0, 0, 3,-1 ]
    # 則1, 2, 3, -1就是低頻處，中間則是高頻處
    # 利用此特性若一取樣後的時域信號x=[1, 2, 3, 4]，經過FFT後可得X = [10, -2+2j, -2, -2-2j]
    # 若oversampling factor = 3
    # 則我們可利用LPF，來進行內插，得到oversampling factor=3的近似取樣信號
    # 將Y = [10, -2+2j, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2-2j]]
    # 中間補0，直到Y有Nfft*L個點
    # 最後將Y做IFFT並乘上L可得內後的結果
    # y = [1 , 0.88+0.43j, 1.38+0.43j, 2, 2.38-0.43j, 2.616-0.43j, 3, 3.61+0.43j, 4.11+0.43j, 4, 3.11-0.43j, 1.88-0.43j]

    # 先在X補0，直到vector長度為Nfft * L[i]   (即為經過LPF)
    oversample_X = X[:Nfft//2] + [0]*(Nfft*L[i] - Nfft) + X[Nfft//2:]

    # 最後做IFFT並乘上L[i]，可得經過LPF內插後的時域信號
    oversample_x = L[i] * np.fft.ifft(oversample_X)

    # 我們可以去找該時域信號的振幅
    # 並找時域信號的實部、虛部
    oversample_x_abs = [0]*len(oversample_x)
    oversample_x_real = [0]*len(oversample_x)
    oversample_x_imag = [0]*len(oversample_x)
    for j in range(len(oversample_x_abs)):
        oversample_x_abs[j] = abs(oversample_x[j])
        oversample_x_real[j] = oversample_x[j].real
        oversample_x_imag[j] = oversample_x[j].imag

    # 我們可以再找其PAPR
    # 找平均功率、和最大功率
    avg_power = 0
    peak_power = 0
    for m in range(len(oversample_x)):
        avg_power += abs(oversample_x[m])**2
        if abs(oversample_x[m])**2 > peak_power:
            peak_power = abs(oversample_x[m])**2
    avg_power /= len(oversample_x)
    PAPR = peak_power / avg_power
    PAPR_dB = 10*np.log10(PAPR)


    # 我們可以畫這Nfft * L[i]個點的實部、虛部scatter圖
    plt.subplot(1,2,1)
    if i == 0:
        scatter1 = plt.scatter(oversample_x_real, oversample_x_imag, c=['red']*len(oversample_x), s=60, marker='x', label='(sampling factor)L={0}'.format(L[i]))
    else:
        scatter2 = plt.scatter(oversample_x_real, oversample_x_imag, c=['blue']*len(oversample_x), s=10, alpha='0.3', label='(sampling factor)L={0}'.format(L[i]))

    # 我們可以注意到當sampling factor = 1 (即沒有發生過取樣oversample時，其時域的Nfft個取樣點其振幅完全相同)
    # 我們可以畫一個半徑=0.25的圓來看到此性質
    angle = [0]*360
    circle_x = [0]*len(angle)
    circle_y = [0]*len(angle)
    for m in range(len(angle)):
        angle[m] = -np.pi + 2*np.pi * m / (len(angle) - 2)
        circle_x[m] = 0.25 * math.cos(angle[m])
        circle_y[m] = 0.25 * math.sin(angle[m])
    plt.plot(circle_x, circle_y, color='red')


    plt.title('scatter of sampling points (time-domain)')
    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.axis('equal')
    plt.axis([-0.4, 0.4, -0.4, 0.4])

    # 並順便畫出時域信號
    plt.subplot(1,2,2)
    if i == 0:
        plt.plot(time[i], oversample_x_abs, color='red', marker='x', label='(sampling factor)L={0}, PAPR={1:0.3f}dB'.format(L[i],PAPR_dB))
    elif i == 1:
        plt.plot(time[i], oversample_x_abs, color='blue', marker='o',label='(sampling factor)L={0}, PAPR={1:0.3f}dB'.format(L[i], PAPR_dB))
    plt.title(r'$IFFT(X_q[k]),\/q=3,\/Nfft={0}$'.format(Nfft))
    plt.xlabel('Time (normalized by symbol duration)')
    plt.ylabel(r'$|IFFT(X_q[k])|$')
    plt.grid(True, which='both')


plt.legend()
plt.show()



