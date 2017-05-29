import numpy as np
import matplotlib.pyplot as plt
import cmath,math

# 以下的模擬為 time-domain estimation techniques for CFO
# 這裡假設STO=0

Nfft = 128                                      # 總共有多少個sub channel
Nusc = 128                                      # 總共有多少sub channel 真正的被用來傳送symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
n_guard = Nfft//4                               # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
N = 1000                                        # 做N次迭代，找估計出來的CFO的MSE
L = 1                                           # 假設有L條multipath
h = [0]*L                                       # 存放multipath通道的 impulase response
H = [0]*N                                       # 為一個list其中有N個元素(因為總共會傳送N個ofdm symbol)，其中每一個元素會對應到一個序列類型的物件，這是代表傳送某個ofdm symbolo時的channel frequency response
max_delay_spread = L-1                          # 最大的 time delay spread為L-1個取樣點的時間，所以會有L條路徑，分別為delay 0,  delay 1,  delay 2 .......delay L-1 時間單位
CFO = [0]*101                                   # 定義CFO (這是normalized frequency offset )
for i in range(len(CFO)):                       # normalize CFO的範圍訂在 -3 ~ 3
    CFO[i] = -3+i*6/(len(CFO)-1)
MSE =[0]*len(CFO)                               # 用來紀錄估計出來的CFO的MSE
snr_db = 30                                     # 決定snr(dB)
snr = np.power(10,snr_db/10)                    # 決定snr


constellation = [-1, 1]                         # 決定星座點
K = int(np.log2(len(constellation)))            # 代表一個symbol含有K個bit


for k in range(3):
    for i in range(len(CFO)):
        mse = 0  # 先將mse歸0

        if k == 1 or k == 2:  # CFO Estimation techniques using training symbol
            # 採用此方法的話要先決定training symbol
            # 先決定training symbol子載波中的repetitive pattern的週期D
            if k == 1:
                D = 2
            elif k == 2:
                D = 4
            Train = [0] * Nfft
            # 假設training symbol的子載波是採用comb-type方式來送symbol
            for m in range(Nfft):  # 假設所有sub-channel 都有送symbol
                if m % D == 0:
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                    for n in range(len(constellation)):
                        if b <= (n + 1) / len(constellation):
                            Train[m] = constellation[n]
                            break
            train = np.fft.ifft(Train)  # 將頻域的Nfft個 symbol 做 ifft 轉到時域

            # 接下來要加上cyclic prefix
            t_new = [0] * (Nfft + n_guard)
            n = 0
            for m in range(Nfft - n_guard, Nfft, 1):
                t_new[n] = train[m]
                n += 1
            for m in range(Nfft):
                t_new[n] = train[m]
                n += 1
            training_symbol = t_new  # 現在training symbol已經有加上cyclic prefix

            # 接下來考慮CFO (carrier frequency offset)造成的影響，相當於乘上np.exp(1j * 2*np.pi * CFO [i] * (m-n_guard) / Nfft)
            # 其中CFO[i] 為normalized frequency offset
            # 注意，一定要加完cp後才能開始考慮CFO造成的影響
            for m in range(len(training_symbol)):
                training_symbol[m] *= np.exp(1j * 2 * np.pi * CFO[i] * (m-n_guard) / Nfft)

        for j in range(N):
            s = []  # 用來存放OFDM symbol

            # 接下來決定OFDM symbol序列
            # 決定所有sub-channel要送哪些信號
            for m in range(Nfft):  # 假設所有sub-channel 都有送symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        X[m] = constellation[n]
                        break

            # 將頻域的Nfft個 symbol 做 ifft 轉到時域
            x = np.fft.ifft(X)

            # 接下來要加上cyclic prefix
            x_new = [0] * (Nfft + n_guard)
            n = 0
            for m in range(Nfft - n_guard, Nfft, 1):
                x_new[n] = x[m]
                n += 1
            for m in range(Nfft):
                x_new[n] = x[m]
                n += 1
            x = x_new  # 現在x已經有加上cyclic prefix

            # 接下來考慮CFO (carrier frequency offset)造成的影響，相當於乘上np.exp(1j * 2*np.pi * CFO [i] * (m-n_guard) / Nfft)
            # 其中CFO[i] 為normalized frequency offset
            # 注意，一定要加完cp後才能開始考慮CFO造成的影響
            for m in range(len(x)):
                x[m] *= np.exp(1j * 2 * np.pi * CFO[i] * (m-n_guard) / Nfft)

            # 這裡先不考慮multipath通道吧

            # 如果是採用training symbol估計CFO的技術，則OFDM symbol序列需事先加上training symbol
            if k == 1 or k == 2:
                s += training_symbol
            # 將這一個OFDM symbol vector x，加到完整的OFDM symbol 序列中
            s += x

            # 決定完OFDM symbol 序列後
            # 接下來要統計平均一個symbol有多少能量
            energy = 0
            for m in range(len(s)):
                energy += abs(s[m]) ** 2

            Es = (energy / len(s)) *  (Nfft+n_guard) / Nfft       # 平均一個取樣點有energy / len(s) 的能量
                                                                  # 而平均一個symbol 的能量為平均一個取樣點能量乘上(Nfft + n_guard) / Nfft
            Eb = Es / K              # 平均一個bit 有Eb的能量
            No = Eb / snr


            # 再來對這個OFDM symbol序列加上雜訊，並估計STO
            s_add_noise = [0]*len(s)
            # 加上Gaussian white noise
            for m in range(len(s_add_noise)):
                s_add_noise[m] = s[m] + np.sqrt(No/2) * np.random.randn() + 1j * np.sqrt(No/2) * np.random.randn()

            # 接下來要估計CFO
            if k == 0: # CFO Estimation Techniques using cyclic prefix (CP)
                total = 0
                for m in range(n_guard):
                    total += s_add_noise[m].conjugate() * s_add_noise[m + Nfft]
                CFO_estimate = 1/(2*np.pi) * cmath.phase(total) # CFO_estimate就是估計出來的結果
                mse += abs(CFO[i] - CFO_estimate)**2
            elif k == 1 or k == 2: # CFO Estimation Techniques using training symbol
                # 先從OFDM symbol序列中取出training symbol部分
                train = s_add_noise[0:(Nfft+n_guard)]

                # 再把training symbol去除CP部分
                train = train[n_guard:]

                total = 0
                for m in range(Nfft//D):
                    total += train[m].conjugate() * train[m + Nfft//D]
                CFO_estimate = D/(2*np.pi) * cmath.phase(total)  # CFO_estimate就是估計出來的結果
                mse += abs(CFO[i] - CFO_estimate)**2

        MSE[i] = mse / N

    if k == 0:
        plt.semilogy(CFO, MSE, label='using CP')
    if k == 1 or k == 2:
        plt.semilogy(CFO, MSE, label='using Training Symbol (D={0})'.format(D))

plt.title('MSE of CFO estimation , SNR={0}dB'.format(snr_db))
plt.legend(loc='upper right')
plt.xlabel('CFO')
plt.ylabel('MSE')
plt.grid(True,which='both')
plt.show()



