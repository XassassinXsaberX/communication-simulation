import numpy as np
import matplotlib.pyplot as plt

# 此模擬的主要目的是要觀察採用raised cosine的pulse shaping並採用DFT-spreading的PAPR 降低技術
# DFT-spreading的技術其實就是將要傳送的symbol vector先做一次precoding
# precoding matrix為DFT matrix
# 所以symbol vector 做一次DFT後，再透過OFDM的調變系統做一次FFT
# 相當於變成 Single - Carrier OFDM
# 接下來我們會將時域序列x[n]，與raised cosine filter h[n]做linear convolution
# 得到的結果即為經過pulse shaping後的結果


Nfft = 256              # 有Nfft個子載波
M = 64                  # 做DFT spreading前，實際上只會用到M個symbol
S = Nfft//M             # bandwidth spreading factor
L = 8                   # oversampling factor
N = 10000               # 做N次來找CCDF
alpha = [0, 0.2, 0.4, 0.6]          # 不同roll-off factor
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

# 定義raised cosine filter
def raised_cosine_filter(N , alpha, Tsym, Ts):
    # 可參考 http://edoras.sdsu.edu/doc/matlab/toolbox/comm/rcosfir.html   https://en.wikipedia.org/wiki/Raised-cosine_filter   https://github.com/veeresht/CommPy/blob/master/commpy/filters.py
    # N代表這個raised cosine filter有多少個取樣點
    # alpha代表roll-off factor
    # Tsym代表symbol period
    # Ts代表取樣間隔

    # 接下來要決定時間序列
    # 每個取樣點的時間間隔為Ts
    # 時間從負到正
    t = [0]*N
    for i in range(N):
        t[i] = i        # 若N = 6，時間序列變 [ 0 , 1 , 2 , 3 , 4 , 5  ]
        t[i] -= N/2     # 若N = 6，時間序列變 [ -3 , -2 , -1 , 0 , 1 , 2  ]
        t[i] *= Ts      # 若N = 6，時間序列變 [ -3Ts , -2Ts , -1Ts , 0 , 1Ts , 2Ts  ]

    # 接下來決定impulse response
    h = [0]*N
    for i in range(N):
        if t[i] == 0:
            h[i] = 1
        elif  alpha != 0 and t[i] == Tsym / (2*alpha):
            h[i] = np.pi/4 * np.sin(np.pi * t[i] / Tsym) / (np.pi * t[i] / Tsym)
        elif  alpha != 0 and t[i] == -Tsym / (2*alpha):
            h[i] = np.pi/4 * np.sin(np.pi * t[i] / Tsym) / (np.pi * t[i] / Tsym)
        else:
            h[i] = np.sin(np.pi * t[i] / Tsym) / (np.pi * t[i] / Tsym) \
                   * np.cos(np.pi * alpha * t[i] / Tsym) / (1 - (4*alpha*alpha*t[i]*t[i]/Tsym/Tsym))

    return h # 傳回係數


# 現在要先做是否採用pulse shaping，及不同roll-off factor的raised cosine filter情況下的模擬
for k in range(2):
    if k == 0:
        constellation = [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]
        constellation_name = 'QPSK'
    elif k == 1:
        constellation = [1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j,
                         -3 - 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j]
        constellation_name = '16QAM'
    plt.figure(k)
    plt.title('{0}'.format(constellation_name))
    plt.xlabel(r'$z^2(dB)$')
    plt.ylabel(r'$CCDF=Probability\/(PAPR\/>\/z^2)$')
    plt.grid(True, which='both')
    for p in range(2):
        if p == 0:  # 採用 Interleaved FDMA的DFT-spreading技術
            # 我們先來做不採用pulse shaping的DFT-spreading情況
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

            plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='IFDMA with no pulse shaping')
            plt.legend()

            # 接下來考慮若採用raised cosine filter來pulse shaping的PAPR
            for q in range(len(alpha)):
                for i in range(len(z_square_dB)):
                    x = [0] * M  # 要送出M個symbol
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
                        # 做FDMA mapping 後變為X_new = [X0 , 0 , 0 , X1 , 0 , 0 , X2 , 0 , 0 , X3 , 0 , 0]

                        # 接下來決定X_new
                        for m in range(len(X_new)):
                            if m % S == 0:
                                X_new[m] = X[m // S]
                            else:
                                X_new[m] = 0

                        # 最後將X_new從頻域轉到時域得到x_new
                        x_new = np.fft.ifft(X_new)

                        # 接下來要對x_new過取樣補0，若L=3
                        # ex 若原本x_new = [x0 , x1 , x2]
                        # 現在變成x_new = [x0 , 0 , 0 , x1 , 0 , 0 , x2 , 0 ,0]
                        x_new2 = [0]*len(x_new)*L
                        for m in range(len(x_new2)):
                            if m % L == 0:
                                x_new2[m] = x_new[m//L]

                        # 接下來要取得raised cosine filter的impulse response
                        h = raised_cosine_filter(63, alpha[q], 1, 1/L)

                        # 最後將x_new2 和 h 做linear convolution得到y即為最後結果
                        y = np.convolve(h, x_new2)

                        PAPR = find_PAPR(y)  # 找出x_new2的PAPR
                        PAPR_dB = 10 * np.log10(PAPR)

                        if PAPR_dB > z_square_dB[i]:
                            count += 1

                    CCDF[i] = count / N

                plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, linestyle='--', label=r'$IFDMA\/with\/\alpha={0}$'.format(alpha[q]))
                plt.legend()


        elif p == 1:  # 採用 Localized FDMA的DFT-spreading技術
            # 我們先來做不採用pulse shaping的DFT-spreading情況
            for i in range(len(z_square_dB)):
                x = [0] * M  # 要送出M個symbol
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

            plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label='LFDMA with no pulse shaping')
            plt.legend()

            # 接下來考慮若採用raised cosine filter來pulse shaping的PAPR
            for q in range(len(alpha)):
                for i in range(len(z_square_dB)):
                    x = [0] * M  # 要送出M個symbol
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

                        # 接下來要對x_new過取樣補0，若L=3
                        # ex 若原本x_new = [x0 , x1 , x2]
                        # 現在變成x_new = [x0 , 0 , 0 , x1 , 0 , 0 , x2 , 0 ,0]
                        x_new2 = [0] * len(x_new) * L
                        for m in range(len(x_new2)):
                            if m % L == 0:
                                x_new2[m] = x_new[m // L]

                        # 接下來要取得raised cosine filter的impulse response
                        h = raised_cosine_filter(63, alpha[q], 1, 1/L)

                        # 最後將x_new2 和 h 做linear convolution得到y即為最後結果
                        y = np.convolve(h, x_new2)

                        PAPR = find_PAPR(y)  # 找出x_new2的PAPR
                        PAPR_dB = 10 * np.log10(PAPR)

                        if PAPR_dB > z_square_dB[i]:
                            count += 1

                    CCDF[i] = count / N

                plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, linestyle='--', label=r'$LFDMA\/with\/\alpha={0}$'.format(alpha[q]))
                plt.legend()

# 最後我們來看看固定roll-off factor，且採用Localized FDMA但不同M時的PAPR
M = [4, 8, 32, 64, 128]
S = [0]*len(M)
for i in range(len(S)):
    S[i] = Nfft // M[i]
alpha = 0.4

plt.figure(2)
plt.title('{0}'.format(constellation_name))
plt.xlabel(r'$z^2(dB)$')
plt.ylabel(r'$CCDF=Probability\/(PAPR\/>\/z^2)$')
plt.grid(True, which='both')
for k in range(len(M)):
    for i in range(len(z_square_dB)):
        x = [0] * M[k]  # 要送出M個symbol
        X_new = [0] * M[k] * S[k] # 實際上會送出的頻域vector
        count = 0
        for j in range(N):
            # 決定所有要送哪些信號
            for m in range(M[k]):
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
                if m < M[k]:
                    X_new[m] = X[m]
                else:
                    X_new[m] = 0

            # 最後將X_new從頻域轉到時域得到x_new
            x_new = np.fft.ifft(X_new)

            # 接下來要對x_new過取樣補0，若L=3
            # ex 若原本x_new = [x0 , x1 , x2]
            # 現在變成x_new = [x0 , 0 , 0 , x1 , 0 , 0 , x2 , 0 ,0]
            x_new2 = [0] * len(x_new) * L
            for m in range(len(x_new2)):
                if m % L == 0:
                    x_new2[m] = x_new[m // L]

            # 接下來要取得raised cosine filter的impulse response
            h = raised_cosine_filter(63, alpha, 1, 1 / L)

            # 最後將x_new2 和 h 做linear convolution得到y即為最後結果
            y = np.convolve(h, x_new2)

            PAPR = find_PAPR(y)  # 找出x_new2的PAPR
            PAPR_dB = 10 * np.log10(PAPR)

            if PAPR_dB > z_square_dB[i]:
                count += 1

        CCDF[i] = count / N

    plt.semilogy(z_square_dB, CCDF, marker='o', lw=1, label=r'$LFDMA\/with\/\alpha={0}\/,\/M={1}$'.format(alpha,M[k]))
    plt.legend()

plt.show()





