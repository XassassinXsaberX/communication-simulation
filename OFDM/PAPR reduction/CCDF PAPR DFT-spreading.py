import numpy as np
import matplotlib.pyplot as plt

# 此模擬的主要目的是要觀察採用DFT-spreading的PAPR 降低技術
# DFT-spreading的技術其實就是將要傳送的symbol vector先做一次precoding
# precoding matrix為DFT matrix
# 所以symbol vector 做一次DFT後，再透過OFDM的調變系統做一次FFT
# 相當於變成 Single - Carrier OFDM


Nfft = 256              # 有Nfft個子載波
M = 64                  # 做DFT spreading前，實際上只會用到M個symbol
S = Nfft//M             # bandwidth spreading factor
N = 10000               # 做N次來找CCDF
z_square_dB = [0]*50                # z^2的dB值
z_square = [0]*len(z_square_dB)     # z^2
z = [0]*len(z_square)               # z
for i in range(len(z_square_dB)):
    z_square_dB[i] = 0 + i*11/(len(z_square_dB) - 1)  # 代表z的平方取dB值
    z_square[i] =  np.power(10,z_square_dB[i]/10)    # 代表z的平方
    z[i] = np.sqrt(z_square[i])                      # 代表z

# 這裡不考慮CP，而且所有子載波都會使用到
CCDF = [0]*len(z_square_dB)       # CCDF模擬結果

# 定義找出PAPR的函數
def find_PAPR(x):
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
    return PAPR


for k in range(3):
    if k == 0:
        plt.subplot(1,3,1)
        constellation = [-1-1j, -1+1j, 1-1j, 1+1j]
        constellation_name = 'QPSK'
    elif k == 1:
        plt.subplot(1,3,2)
        constellation =  [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
        constellation_name = '16QAM'
    elif k == 2:
        plt.subplot(1,3,3)
        constellation = []
        set_value = [-7, -3, -5, -1, 1, 3, 5, 7]
        for i in range(len(set_value)):
            for j in range(len(set_value)):
                constellation += [set_value[i] + 1j * set_value[j]]
        constellation_name = '64QAM'

    plt.title('{0}'.format(constellation_name))
    plt.xlabel(r'$z^2(dB)$')
    plt.ylabel(r'$CCDF=Probability\/(PAPR\/>\/z^2)$')
    plt.grid(True, which='both')

    for p in range(3):
        # 未採用PAPR reduction時的情況
        if p == 0:
            for i in range(len(z_square_dB)):
                X = [0]*Nfft            # 要送出Nfft個symbol
                count = 0
                for j in range(N):
                    # 決定所有sub-channel要送哪些信號
                    for m in range(Nfft):
                        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                        for n in range(len(constellation)):
                            if b <= (n + 1) / len(constellation):
                                X[m] = constellation[n]
                                break

                    x = np.fft.ifft(X)   # 作IFFT頻域轉到時域

                    PAPR = find_PAPR(x)     # 找出x的PAPR
                    PAPR_dB = 10 * np.log10(PAPR)


                    if PAPR_dB > z_square_dB[i]:
                        count += 1

                CCDF[i] = count / N

            plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='OFDMA')
            plt.legend()

        # 採用 Interleaved FDMA的DFT-spreading技術
        elif p == 1:
            for i in range(len(z_square_dB)):
                x = [0]*M           # 要送出M個symbol
                X_new = [0]*M*S     # 實際上會送出的頻域vector
                count = 0
                for j in range(N):
                    # 決定所有要送哪些信號
                    for m in range(M):
                        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                        for n in range(len(constellation)):
                            if b <= (n + 1) / len(constellation):
                                x[m] = constellation[n]
                                break

                    X = np.fft.fft(x) # 先對原本要送出去的symbol vector做DFT spreading

                    # 接下來再做FDMA mapping
                    # ex 若 M = 4 , S = 3 , N = M * S = 12
                    # X = [X0 , X1 , X2 , X3]
                    # 做FDMA mapping 後變為X_new = [X0 , 0 , 0 , X1 , 0 , 0 , X2 , 0 , 0 , X3 , 0 , 0]

                    # 接下來決定X_new
                    for m in range(len(X_new)):
                        if m % S == 0:
                            X_new[m] = X[m//S]
                        else:
                            X_new[m] = 0

                    # 最後將X_new從頻域轉到時域得到x_new
                    x_new = np.fft.ifft(X_new)

                    PAPR = find_PAPR(x_new)  # 找出x_new的PAPR
                    PAPR_dB = 10 * np.log10(PAPR)

                    if PAPR_dB > z_square_dB[i]:
                        count += 1

                CCDF[i] = count / N

            plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='IFDMA')
            plt.legend()

        # 採用 Localized FDMA的DFT-spreading技術
        elif p == 2:
            for i in range(len(z_square_dB)):
                x = [0] * M          # 要送出M個symbol
                X_new = [0] * M * S  # 實際上會送出的頻域vector
                count = 0
                for j in range(N):
                    # 決定所有要送哪些信號
                    for m in range(M):
                        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                        for n in range(len(constellation)):
                            if b <= (n + 1) / len(constellation):
                                x[m] = constellation[n]
                                break

                    X = np.fft.fft(x)  # 先對原本要送出去的symbol vector做DFT spreading

                    # 接下來再做FDMA mapping
                    # ex 若 M = 4 , S = 3 , N = M * S = 12
                    # X = [X0 , X1 , X2 , X3]
                    # 做FDMA mapping 後變為X_new = [X0 , X1 , X2 , X3 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]

                    # 接下來決定X_new
                    for m in range(len(X_new)):
                        if m < M:
                            X_new[m] = X[m]
                        else:
                            X_new[m] = 0

                    # 最後將X_new從頻域轉到時域得到x_new
                    x_new = np.fft.ifft(X_new)

                    PAPR = find_PAPR(x_new)  # 找出x_new的PAPR
                    PAPR_dB = 10 * np.log10(PAPR)

                    if PAPR_dB > z_square_dB[i]:
                        count += 1

                CCDF[i] = count / N

            plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='LFDMA')
            plt.legend()

plt.show()