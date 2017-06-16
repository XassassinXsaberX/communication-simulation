import numpy as np
import matplotlib.pyplot as plt

# 此模擬的目的是要觀察採用PTS (Partial Transmit Sequence)，這種PAPR reduction coding技術
# 是否會真的使PAPR降低(我們可透過CCDF來觀察PAPR)
# 注意：本次模擬沒有採用過取樣(oversampling)！

Nfft = 256                          # 有Nfft個子載波
X = [0]*Nfft                        # 傳送Nfft個symbol
N = 100000                          # 做N次來找CCDF
V = [1, 2, 4, 8 ,16]                # 將symbol vector分成V[i] 個subblock
z_square_dB = [0]*50                # z^2的dB值
z_square = [0]*len(z_square_dB)     # z^2
z = [0]*len(z_square)               # z
for i in range(len(z_square_dB)):
    z_square_dB[i] = 4 + i*7/(len(z_square_dB) - 1)  # 代表z的平方取dB值
    z_square[i] =  np.power(10,z_square_dB[i]/10)    # 代表z的平方
    z[i] = np.sqrt(z_square[i])                      # 代表z

# 這裡不考慮CP，而且所有子載波都會使用到
CCDF = [0]*len(z_square_dB)       # CCDF模擬結果

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


# 首先考慮未採用PAPR reduction技術時，OFDM signal的CCDF
for i in range(len(z_square_dB)):
    CCDF[i] = 0
    count = 0
    for j in range(N):
        # 決定所有sub-channel要送哪些信號
        for m in range(Nfft):
            b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
            for n in range(len(constellation)):
                if b <= (n + 1) / len(constellation):
                    X[m] = constellation[n]
                    break

        x = np.fft.ifft(X) * np.sqrt(Nfft / Es)   # 作IFFT頻域轉到時域
        # 乘上np.sqrt(Nfft / Es)的用意是將每個取樣點的能量normalize成1

        # 接下來要找PAPR(Peak-to-Average Power Ratio)
        # 要找出x[n] power的平均值
        avg_power = 0
        for m in range(len(x)):
            avg_power += abs(x[m]) ** 2
        avg_power /= len(x)

        # 再找出x[n]中最大的功率
        peak_power = 0
        for m in range(len(x)):
            if abs(x[m]) ** 2 > peak_power:
                peak_power = abs(x[m]) ** 2
        # 有了Peak power及Average power即可求PAPR
        PAPR = peak_power / avg_power
        PAPR_dB = 10 * np.log10(peak_power / avg_power)

        if PAPR_dB > z_square_dB[i]:
            count += 1

    CCDF[i] = count / N

plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='original')

# 接下來考慮採用Partial Transmit Sequence的降低PAPR技術

for k in range(len(V)):
    for i in range(len(z_square_dB)):
        CCDF[i] = 0
        count = 0
        for j in range(N):
            # 決定所有sub-channel要送哪些信號
            for m in range(Nfft):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break


            # 以下為MIMO-OFDM wireless communication with matlab中的suboptimal combination algorithm

            # 先從input symbol vector X中，建立V[k]個不相交的subblocks
            # 可參考MIMO-OFDM wireless communication with matlab 式(7.26)
            subblock = [0]*V[k] # 接下來每個subblock的元素都存放一個具有Nfft個元素的vector
            tmp = 0
            for m in range(V[k]):
                tmp_vector = [0]*Nfft
                for n in range(Nfft//V[k]):
                    tmp_vector[tmp] = X[tmp]
                    tmp += 1
                subblock[m] = np.array(tmp_vector)

            phase_factor = [1]*V[k] # phase factor中有V[k]個元素
            # 現在要依序決定phase factor中的每個元素為1還是-1

            # 首先要找出PAPR
            x = np.fft.ifft(X)
            # 要找出x[n] power的平均值
            avg_power = 0
            for m in range(len(x)):
                avg_power += abs(x[m]) ** 2
            avg_power /= len(x)

            # 再找出x[n]中最大的功率
            peak_power = 0
            for m in range(len(x)):
                if abs(x[m]) ** 2 > peak_power:
                    peak_power = abs(x[m]) ** 2
            # 有了Peak power及Average power即可求PAPR
            PAPR = peak_power / avg_power
            PAPR_min = PAPR   # 將此PAPR設為PAPR_min

            for m in range(1,V[k],1):
                phase_factor[m] = -1
                # 找出在此phase_factor的情況下的PAPR
                code_X = np.array([0j]*Nfft)
                for n in range(V[k]):
                    code_X += subblock[n]*phase_factor[n]
                # code_X為在此phase_factor的情況下，將symbol vector X編碼後的結果 (可參考MIMO-OFDM wireless communication with matlab 式(7.27) )
                # 接著要找出code_X對應到的PAPR
                # 先轉到時域
                code_x = np.fft.ifft(code_X)

                # 要找出code_x[n] power的平均值
                avg_power = 0
                for n in range(len(code_x)):
                    avg_power += abs(code_x[n]) ** 2
                avg_power /= len(code_x)

                # 再找出code_x[n]中最大的功率
                peak_power = 0
                for n in range(len(code_x)):
                    if abs(code_x[n]) ** 2 > peak_power:
                        peak_power = abs(code_x[n]) ** 2
                # 有了Peak power及Average power即可求PAPR
                PAPR = peak_power / avg_power

                if PAPR < PAPR_min:
                    PAPR_min = PAPR
                else:
                    phase_factor[m] = 1

            PAPR_min_dB = 10*np.log10(PAPR_min)
            if PAPR_min_dB > z_square_dB[i]:
                count += 1

        CCDF[i] = count / N

    plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='V={0}'.format(V[k]))


plt.xlabel(r'$z^2(dB)$')
plt.ylabel(r'$CCDF=Probability\/(PAPR\/>\/z^2)$')
plt.legend()
plt.grid(True,which='both')
plt.show()


