import numpy as np
import matplotlib.pyplot as plt

freq_offset = [0]*41                            # 用來紀錄frequency offset  單位Hz( freq_offset[k] = delta / T_symbol  )
freq_offset_kHz = [0]*len(freq_offset)          # 用來紀錄frequency offset  單位kHz
delta = [0]*len(freq_offset)                    # delta[i] = freq_offset[i] * T_symbol
error = [0]*len(freq_offset)                    # 用來紀錄symbol error的程度
theory_error = [0]*len(freq_offset)             # 理論上的symbol error的程度
Nfft = 64                                       # 總共有多少個sub channel
Nusc = 52                                       # 總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
n_guard = 16                                    # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
N = 1                                           # 做N次迭代來找error
snr_dB = 30                                     # snr設定成30dB
snr = np.power(10,snr_dB/10)

constellation = [-1,1] # 決定星座點
K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量
Eb_old = Eb

# 實際的頻率偏移量為freq_offset = [ -200kHz , -190kHz , -180kHz ...... 180kHz , 190kHz , 200kHz ]
# 頻率偏移的程度，此例為delta = [-200 , -190 , -180 ...... 180 , 190 , 200]
for i in range(len(freq_offset)):
    freq_offset_kHz[i] = -200 + 10*i
    freq_offset[i] = (-200 + 10*i)*1000     # 實際的頻率偏移值
    delta[i] = freq_offset[i] * T_symbol    # 決定delta值


for k in range(2):
    for j in range(len(freq_offset)):
        error[j] = 0
        for i in range(N):
            # 決定所有sub-channel要送哪些信號
            for m in range(64):  # 假設sub-channel : 0,1,2,3,4,5及58,59,60,61,62,63不用來傳送symbol
                if m >= 6 and m <= 57:
                    X[m] = constellation[1]  # 假設都送出相同的symbol
                else:
                    X[m] = 0

            # 將頻域的Nfft個 symbol 做 ifft 轉到時域
            x = np.fft.ifft(X) * Nfft / np.sqrt(Nusc) / np.sqrt(Es)
            # 乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)可將一個OFDM symbol的總能量normalize成 1* Nfft  (從時域的角度來看)
            # 你可以這樣想，送了 Nusc = 52 個symbol，總能量卻為 Nfft = 64，所以平均每symbol花了(64 / 52)的能量-----------------------------------------------------------(式1)
            # 而在時域總共有64個取樣點，所以平均每個取樣點的能量為1

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

            # 接下來考慮頻率偏移的情況
            #T_symbol = 20 * (10 ** 6)
            #t_sample = t_sample = 10**3
            for m in range(len(x)):
                x[m] *= np.exp(1j * 2*np.pi * freq_offset[j] * (m+1)*t_sample) # 頻率偏移 ( freq_offset[j] / T_symbol ) Hz，此時的時間為(m+1)*t_sample


            y = x
            ######################################################################################################
            # 以上為傳送端
            #
            # 現在將傳送出去的OFDM symbol加上雜訊
            # Eb代表的是平均每個bit的能量
            Eb = 1 / K * (Nfft / Nusc) * ((Nfft + n_guard) / Nfft)
            # 若原本一個symbol 能量Es = 1，現在變成1 * (Nfft / Nusc) 可見(式1) ，現在一個取樣點平均能量為1
            # 若原本一個取樣點平均能量為1，加上cp後變成要乘上((Nfft+n_guard) / Nfft)
            # 因為一個symbol含K個bit (所以Eb = Es / K)，這就是為何要除上K
            No = Eb / snr
            for m in range(Nfft + n_guard):
                y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
            #
            # 以下為接收端
            ######################################################################################################

            # 接下來要對接收向量去除cyclic prefix
            y_new = [0] * Nfft
            n = 0
            for m in range(n_guard, Nfft + n_guard, 1):
                y_new[n] = y[m]
                n += 1
            y = y_new  # 現在y已經去除OFDM的cyclic prefix

            for m in range(len(y)):
                y[m] = y[m] * np.sqrt(Es) * np.sqrt(Nusc) / Nfft  # 因為前面x向量有乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)，所以現在要變回來
            Y = np.fft.fft(y)  # 現在將y轉到頻域，變成Y

            # 接下來判斷理論上ICI導致symbol error的程度
            if k == 0:
                Y_theory = [0]*Nfft
                for n in range(Nfft):
                    Y_theory[n] = 0
                    for m in range(Nfft):
                        if X[m] != 0 and (np.exp(1j*2*np.pi*(m+delta[j]-n))-1) == 0 and (1j*2*np.pi*(m+delta[j]-n))==0:
                            Y_theory[n] += X[m]
                        elif X[m] == 0:
                            Y_theory[n] += 0
                        else:
                            Y_theory[n] += X[m] * (np.exp(1j*2*np.pi*(m+delta[j]-n))-1) / (1j*2*np.pi*(m+delta[j]-n))
                    error[j] += abs(Y_theory[n] - X[n]) ** 2

            # 接下來判斷實際上ICI導致symbol error的程度
            elif k == 1:
                for m in range(Nfft):
                    error[j] += abs(Y[m]-X[m])**2

        error[j] = error[j] / (Nfft*N)      # error[j] 是代表平均一個sub channel的symbol與原本的symbol發生了多少誤差
        error[j] = 10*np.log10(error[j])    # 取其dB值

    if k == 0:
        error[len(error) // 2] = -snr_dB + 10 * np.log10(Eb_old)
        # 理論上在沒有雜訊的情況，當頻率偏移量為0時，是毫無錯誤的，所以其error magnitude的dB值為 負無窮大
        # 但考慮有雜訊的情況下，其error magnitude的dB值應為上述公式
        plt.plot(freq_offset_kHz, error, marker='o', label='theory')
    elif k == 1:
        plt.plot(freq_offset_kHz, error, marker='o', label='simulation')
        #plt.ylim(min(error) - 5, max(error) + 5)

plt.title('Error magnitude with frequency offset')
plt.legend()
plt.ylabel('Error (dB)')
plt.xlabel('frequency offset, kHz')
plt.grid(True,which='both')
plt.show()


