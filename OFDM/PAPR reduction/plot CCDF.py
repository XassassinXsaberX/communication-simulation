import numpy as np
import matplotlib.pyplot as plt

z_square_dB = [0]*21
z_square = [0]*len(z_square_dB)
z = [0]*len(z_square_dB)
for i in range(len(z_square_dB)):
    z_square_dB[i] = 4+0.3*i                        # 代表z的平方取dB值
    z_square[i] = np.power(10,z_square_dB[i]/10)    # 代表z的平方
    z[i] = np.sqrt(z_square[i])                     # 代表z


Nfft = [64, 128, 256, 512, 1024]             # 有不同的Nfft點的OFDM symbol
# 這裡不考慮CP，而且所有子載波都會使用到
CCDF_theory = [0]*len(z_square_dB)           # CCDF理論值
CCDF_simulation = [0]*len(z_square_dB)       # CCDF模擬結果
N = 1000                                     # 做N次來取模擬值

# 定義M-PSK的星座點。注意當M=1，即BPSK時的模擬值會與理論值不同
# 補充一下理論值不因星座點而變，但模擬值似乎會因是否採用BPSK而變
# 似乎是因為採用BPSK的PAPR會比較小的原因，而採用QPSK或8-PSK以上的PSK調變，PAPR就不會變小
# M>=2時就沒問題了
constellation = [-1, 1]                               # BPSK星座點的模擬結果會出問題
constellation = [-1-1j, -1+1j, 1-1j, 1+1j]            # QPSK星座點的模擬結果就與理論值相當吻合
#constellation =  [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
# 採用16-QAM亦可得到正確的模擬結果

K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量

for k in range(len(Nfft)):
    X = [0]*Nfft[k]    # 有Nfft[k]個子載波
    for i in range(len(z_square_dB)):
        count = 0      # 用來統計CCDF
        var = [0]*N    # 用來記錄每一次估計出來的var
        for j in range(N):
            # 決定所有sub-channel要送哪些信號
            for m in range(Nfft[k]):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break

            x = np.fft.ifft(X)*np.sqrt(Nfft[k] / Es)  # 作IFFT頻域轉到時域
            # 我們已知Es = E[  | X[k] | ^ 2  ]
            # 經過IFFT後 E[  | x[n] | ^ 2  ] = ES / Nfft (請見MIMO-OFDM wireless communication p176的數學推導)
            # 所以乘上sqrt(Nfft[k] / Ex) 的用意是將 E[  | x[n] | ^ 2  ] normalized成1

            # 當Nfft越大，根據中央極限定理，x的實部及虛部會越接近高斯分布
            # 我們接下來要估計x的實部的variance
            # 首先要求出x的實部的sample mean
            avg_x = 0
            for m in range(len(x)):
                avg_x += x[m].real
            avg_x /= len(x)  # x的實部的sample mean

            # 接下來利用剛剛求出來的sample mean求variance的估計值
            for m in range(len(x)):
                var[j] += abs(avg_x - x[m].real)**2
            var[j] /= (len(x) - 1) # unbiased estrimator of variance

            # 接下來要找PAPR(Peak-to-Average Power Ratio)
            # 要找出x[n] power的平均值
            avg_power = 0
            for m in range(len(x)):
                avg_power += abs(x[m])**2
            avg_power /= len(x)

            # 再找出x[n]中最大的功率
            peak_power = 0
            for m in range(len(x)):
                if abs(x[m])**2 > peak_power:
                    peak_power = abs(x[m])**2
            # 有了Peak power及Average power即可求PAPR
            PAPR = peak_power / avg_power
            PAPR_dB = 10*np.log10(peak_power / avg_power)

            # 最後我們判斷PAPR(dB)是否大於z平方(dB)
            if PAPR_dB > z_square_dB[i]:
                count += 1

        # 對N個variance估計值取sample mean
        variance = 0
        for m in range(len(var)):
            variance += var[m]
        variance /= len(var)

        # 有了variance及z，我們可以利用公式求出CCDF (可參考MIMO-OFDM wireless communication with MATLAB 式(7.9)左方中我的公式推導)
        CCDF_theory[i] = 1 - np.power(1 - np.exp(-z[i]*z[i]/(2*variance)) , Nfft[k])
        #CCDF_theory[i] = 1 - np.power(1 - np.exp(-z[i]*z[i]), Nfft[k])

        # 我們也可以利用剛剛的count次數來統計CCDF的模擬值
        CCDF_simulation[i] = count / N

    plt.semilogy(z_square_dB, CCDF_theory, label='ideal (N={0})'.format(Nfft[k]))
    plt.semilogy(z_square_dB, CCDF_simulation, marker='o', linestyle='--', label='simulation (N={0})'.format(Nfft[k]))

plt.xlabel(r'$z^2(dB)$')
plt.ylabel(r'$CCDF=Probability\/(PAPR\/>\/z^2)$')
plt.legend()
plt.grid(True,which='both')
plt.show()










