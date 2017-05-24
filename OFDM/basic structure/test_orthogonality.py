import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# 這個模擬的目的是為了判斷信號的正交性
# 當信號中出現不連續或是兩信號的頻率差不等於 k / T_symbol ( k為整數、T_symbol為一個symbol的時域周期 )
# 則內積不為0，不正交
# 我們也可以發現信號的delay，不影響正交性

# 而且還有很重要的一點，給定T_symbol(symbol 周期)時，我們會得到子載波的頻率為 k / T_symbol , k = 1,2,3....
# 若取樣間隔為Ts，且Ts * Nfft = T_symbol (也就是一個symbol 周期內取樣Nfft個點)
# 則只有剛好取Nfft個離散時域序列其FFT的離散頻譜才有正交性，取大於Nfft或小於Nfft個點就沒有正交性了


Nfft = 16                      # 做Nfft的點數
T_symbol = 1.6                 # 一個symbol的周期，這也代表其他子載波的頻率為( k / T_symbol )Hz , k = 1,2,3...

t = [0]*1000
T = 0.002  # 連續信號的間隔
for i in range(len(t)):
    t[i] = T*i
x = [0]*len(t)
re_x = [0]*len(t)

Ts = T_symbol / Nfft            # 取樣周期不可為其他數，否則會出現非正交
t_sample = [0]*(int(2/Ts)+1)    # 總共會取樣(int(2/Ts)+1)個點
for i in range(len(t_sample)):
    t_sample[i] = Ts*i
x_sample = [0]*len(t_sample)
re_x_sample = [0]*len(t_sample)

# 共取樣(int(2/Ts)+1)個點，選前Nfft個點
f_index = [0]*Nfft
for i in range(len(f_index)):
    f_index[i] = i

x_matrix = [[0j]*6 for i in range(Nfft)]
x_matrix = np.matrix(x_matrix)



# 先畫出連續信號
for k in range(6):
    for i in range(len(t)):
        if k == 0:
            x[i] = np.exp(1j * 2*np.pi * t[i] / T_symbol)                   # 頻率為 ( 1/T_symbol ) Hz、exponential 信號delay 0秒
            f = 1
            delay = 0
        elif k == 1:
            x[i] = np.exp(1j * 2*np.pi * 2 * t[i] / T_symbol)               # 頻率為 ( 2/T_symbol ) Hz、exponential 信號delay 0秒
            f = 2
            delay = 0
        elif k == 2:
            x[i] = np.exp(1j * 2*np.pi * 3 * (t[i] - 0.1) / T_symbol)       # 頻率為 ( 3/T_symbol ) Hz、exponential 信號delay 0.1秒
            f = 3
            delay = 0.1
        elif k == 3:
            x[i] = np.exp(1j * 2*np.pi * 4 * (t[i] - 0.7) / T_symbol)       # 頻率為 ( 4/T_symbol ) Hz、exponential 信號delay 0.7秒
            f = 4
            delay = 0.7
        elif k == 4:
            x[i] = np.exp(1j * 2*np.pi * 3.5 * t[i] / T_symbol)             # 頻率為 ( 3.5/T_symbol ) Hz、exponential 信號delay 0秒
            f = 3.5
            delay = 0
        elif k == 5:
            f1 = 4
            delay1 = 0.3
            f2 = 3
            delay2 = 0.7
            if t[i] <= 1.3:
                x[i] = np.exp(1j * 2*np.pi * 4 * (t[i] - 0.3) / T_symbol)   # 頻率為 ( 4/T_symbol ) Hz、exponential信號delay 0.3秒(不連續信號)
            else:
                x[i] = np.exp(1j * 2*np.pi * 3 * (t[i] - 0.7) / T_symbol)

    for j in range(len(t)):
        re_x[j] = x[j].real

    plt.subplot(6, 2, 2*k + 1)
    if k == 5:
        plt.title(r'$when\/\/t<1.3\/\/cos(   \frac{{   j2\pi\cdot{0}\cdot (t-{1})   }}    {{  {2}  }}  ),\/\/else\/\/cos(   \frac{{   j2\pi\cdot{3}\cdot (t-{4})   }}    {{  {2}  }}  )$'.format(f1, delay1, T_symbol, f2, delay2))
    else:
        plt.title(r'$cos(   \frac{{   j2\pi\cdot  {0}  \cdot (t  -  {1} )   }}    {{  {2}  }}  )$'.format(f,delay,T_symbol))
    plt.plot(t, re_x ,linestyle = '--')
    plt.xlim(0,2)

# 接著畫出取樣後的信號
for k in range(6):
    for i in range(len(t_sample)):
        # 這6個exponential 信號中，只有前4個會互相正交
        if k == 0:
            x_sample[i] = np.exp(1j * 2*np.pi * t_sample[i] / T_symbol)                     # 頻率為 ( 1/T_symbol ) Hz、exponential 信號delay 0秒
        elif k == 1:
            x_sample[i] = np.exp(1j * 2*np.pi * 2 * t_sample[i] / T_symbol)                 # 頻率為 ( 2/T_symbol ) Hz、exponential 信號delay 0秒
        elif k == 2:
            x_sample[i] = np.exp(1j * 2*np.pi * 3 * (t_sample[i] - 0.1) / T_symbol)         # 頻率為 ( 3/T_symbol ) Hz、exponential 信號delay 0.1秒
        elif k == 3:
            x_sample[i] = np.exp(1j * 2*np.pi * 4 * (t_sample[i] - 0.7) / T_symbol)         # 頻率為 ( 4/T_symbol ) Hz、exponential 信號delay 0.7秒
        elif k == 4:
            x_sample[i] = np.exp(1j * 2*np.pi * 3.5 * t_sample[i] / T_symbol)               # 頻率為 ( 3.5/T_symbol ) Hz、exponential 信號delay 0秒
        elif k == 5:
            if t_sample[i] <= 1.3:
                x_sample[i] = np.exp(1j * 2*np.pi * 4 * (t_sample[i] - 0.3) / T_symbol)     # 頻率為 ( 4/T_symbol ) Hz、exponential 信號delay 0秒(不連續信號)
            else:
                x_sample[i] = np.exp(1j * 2*np.pi * 3 * (t_sample[i] - 0.7) / T_symbol)

        if i < 16:
            x_matrix[i,k] = x_sample[i]

    for j in range(len(x_sample)):
        re_x_sample[j] = x_sample[j].real

    X = np.fft.fft(x_sample[0:Nfft]) #從(int(2/Ts)+1)個取樣點中，選前Nfft個點來做FFT
    for j in range(len(X)):
        X[j] = abs(X[j])

    plt.subplot(6, 2, 2*k+1)
    plt.stem(t_sample,re_x_sample)
    plt.ylabel('Re{x[n]}').set_rotation(0)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.08, 0.9)  # 用來設定座標軸名稱的位置
    plt.xlabel('t')
    ax.xaxis.set_label_coords(1.05, -0.025)  # 用來設定座標軸名稱的位置

    plt.subplot(6, 2, 2*k+2)
    plt.stem(f_index,X)
    plt.ylabel('Re{X[k]}').set_rotation(0)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.08,0.9) #用來設定座標軸名稱的位置
    plt.xlim(0,Nfft)
    plt.xticks(np.arange(0,Nfft,2))

ans = x_matrix.getH() * x_matrix / (T_symbol/Ts) #可觀察ans matrix來判斷信號間是否有正交性
print(ans)
# 結果表明只有前4個信號互相正交
plt.show()


