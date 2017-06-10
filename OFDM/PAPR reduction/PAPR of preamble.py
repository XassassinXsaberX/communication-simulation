import numpy as np
import matplotlib.pyplot as plt
import math

# 此模擬的主要目的是觀察 IEEE 802.16e preamble的PAPR
# IEEE 802.16e 定義了114個preamble
# 我們會看看這114個preamble在不同的取樣速率下的PAPR為何

L = [1,4]             # oversampling factor 分別為1和4，當oversampling factor > 1時，代表過取樣(oversample)
PAPR_dB = [0]*114     # 代表不同的preamble會對應到何種PAPR(dB)
index = [0]*114       # 代表哪一個preamble
for i in range(114):
    index[i] = i

for k in range(len(L)):
    for i in range(114):
        X = []  # 代表每個子載波存放的symbol
        # 接著開始進行讀檔
        with open('./Wibro-Preamble/Preamble_sym{0}.dat'.format(i)) as f:
            while True:
                line = f.readline() # 每次讀出一行
                if line == '': # 若讀到空字串，就結束讀檔
                    break
                symbol = float(line.split('\t')[0])  # 將讀到的字串分開，並變成float物件
                X += [symbol]

        # 讀檔完成後，我們有了1024個的子載波所對應到的每個symbol

        # 我們會先對每個子載波經過sgn函數
        for m in range(len(X)):
            if X[m] > 0:
                X[m] = 1
            elif X[m] < 0:
                X[m] = -1

        # 做一次FFT SHIFT
        # 也就是說針對頻域的vector，將DC分量移到頻譜中心,重新排列fft
        # ex 原本是 X = [1, 2, 3, 4]
        # 現在變為X = [3, 4, 1, 2]
        # 或是原本為 X = [1, 2, 3, 4, 5]
        # 現在變為X = [4, 5, 1, 2, 3]
        X = X[(len(X)+1)//2:] + X[:(len(X)+1)//2]

        # 接著將其轉到時域
        # 我們會利用LPF的方法，依據oversampling factor來決定要補多少零再高頻處，最後在做IFFT轉成時域信號x
        # 具體說明可參考 PAPR of Chu sequence  (第27行開始的註解)

        #  先在X的高頻處補0
        X = X[0:len(X)//2] + [0]*(len(X)*L[k] - len(X)) + X[len(X)//2:]

        # 最後做IFFT並乘上L[k]，可得經過LPF內插後的時域信號
        oversample_x = L[k] * np.fft.ifft(X)

        # 在來統計此時的PAPR
        # 找平均功率、和最大功率
        avg_power = 0
        peak_power = 0
        for m in range(len(oversample_x)):
            avg_power += abs(oversample_x[m]) ** 2
            if abs(oversample_x[m]) ** 2 > peak_power:
                peak_power = abs(oversample_x[m]) ** 2
        avg_power /= len(oversample_x)
        PAPR = peak_power / avg_power
        PAPR_dB[i] = 10 * np.log10(PAPR)

    if k == 0:
        plt.plot(index, PAPR_dB, marker='x', color='red',label='(sampling factor)L={0}'.format(L[k]))
    elif k == 1:
        plt.plot(index, PAPR_dB, marker='o', color='blue',label='(sampling factor)L={0}'.format(L[k]))

plt.title('IEEE 802.16e preamble')
plt.xlabel('Preamble index [0~113]')
plt.ylabel('PAPR(dB)')
plt.legend()
plt.show()




