import numpy as np
import matplotlib.pyplot as plt
import math

#reference : http://www.dsplog.com/2008/08/26/ofdm-rayleigh-channel-ber-bpsk/

snr_db = [0]*15
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 10000              #執行N次來找錯誤率
Nfft = 64               #總共有多少個sub channel
X = [0]*64              #從頻域送出64
Nusc = 52               #總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
t_sample = 5*10**(-8)   #取樣間隔
n_guard = 16            #經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
L = 10                  #假設有L條path
h = [0]*L               #每個元素代表L條path各自的impulse response  (因為在一個OFDM symbol period內為常數，所以很明顯是非時變通道)

#constellation = [1+1j, 1-1j, -1+1j, -1-1j] # 決定QPSK星座點
#constellation_name = 'QPSK'
constellation =  [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j] # 決定16-QAM的16個星座點
constellation_name = '16-QAM'
#constellation = [-1,1] # 決定BPSK星座點
#constellation_name = 'BPSK'

K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量

#當ofdm symbol發生delay一個單位時間，相當於一個往右shift
def shift(x):
    x_new = [0]*len(x)
    for m in range(1,len(x),1):
        x_new[m] = x[m-1]
    return x_new


for i in range(len(snr_db)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(2):
    for i in range(len(snr)):
        error = 0
        if k==0 and (constellation_name == 'BPSK' or constellation_name == 'QPSK'): # SISO - rayleigh (BPSK) (theory)
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue
        elif k==0 and constellation_name == '16-QAM': # SISO - approximation (16-QAM) (theory)
            a = 2 * (1 - 1 / K) / np.log2(K)
            b = 6 * np.log2(K) / (K * K - 1)
            rn = b * snr[i] / 2
            ber[i] = 1 / 2 * a * (1 - np.sqrt(rn / (rn + 1)))
            continue
        for j in range(N):
            if k==1 :
                #決定所有sub-channel要送哪些信號
                for m in range(64):#假設sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
                    if (m>2 and m<29) or (m>34 and m<61):
                        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                        for n in range(len(constellation)):
                            if b <= (n + 1) / len(constellation):
                                X[m] = constellation[n]
                                break
                    else:
                        X[m] = 0

                # 將頻域的Nfft個 symbol 做 ifft 轉到時域
                x = np.fft.ifft(X) * Nfft / np.sqrt(Nusc) / np.sqrt(Es)
                # 乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)可將一個OFDM symbol的總能量normalize成 1* Nfft  (從時域的角度來看)
                # 你可以這樣想，送了 Nusc = 52 個symbol，總能量卻為 Nfft = 64，所以平均每symbol花了(64 / 52)的能量-----------------------------------------------------------(式1)
                # 而在時域總共有64個取樣點，所以平均每個取樣點的能量為1


                #接下來要加上cyclic prefix
                x_new = [0]*(Nfft+n_guard)
                n = 0
                for m in range(Nfft-n_guard,Nfft,1):
                    x_new[n] = x[m]
                    n += 1
                for m in range(Nfft):
                    x_new[n] = x[m]
                    n += 1
                x = x_new  #現在x已經有加上cyclic prefix


                #接下來產生L條path，且假設每一條路徑在一個OFDM symbol 周期內為常數  (所以coherence time > OFDM symbol period)
                y = [0]*(Nfft+n_guard)  #先將接收端會收到的向量清空
                for m in range(L):
                    h[m] = 1/np.sqrt(2)/np.sqrt(L)*np.random.randn() + 1j/np.sqrt(2)/np.sqrt(L)*np.random.randn()#產生一個非時變通道
                    #h[m] 除上 np.sqrt(L) 是為了要normalize multipath channel 的平均功率成1
                    for n in range(Nfft+n_guard):
                        y[n] += x[n]*h[m]
                    x = shift(x) # OFDM symbol往右位移一時間單位，相當於做一次shift
                # y[0] ~ y[Nfft+n_guard] 即為經過Rayleigh fading後的OFDM symbol  (尚未加上noise)

                H = np.fft.fft(h,64)  # H即為multipath channel 的frequency response (有64個點)
                H = H*np.sqrt((Nfft + n_guard) / Nfft) #注意，因為之後y向量的每個元素會乘上np.sqrt((Nfft + n_guard) / Nfft)，所以通道的頻率響應也要跟著乘上np.sqrt((Nfft + n_guard) / Nfft)才行


                # 接下來將每個x的元素乘上(Nfft+n_guard)/Nfft，代表考慮加上cyclic prefix後多消耗的能量
                # 如果沒有乘上(Nfft+n_guard)/Nfft ， 那麼有加cyclic prefix 跟沒加cyclic prefix，每個取樣點所耗去的能量都是相同的-->不合理
                # ex
                # 若時域信號向量Nfft = 4個點為 [ 1, -1, 1, 1 ]
                # 加上cyclic prefix後變為Nfft個點為 [ 1, -1, 1, 1 ]   Ng = 1個點，所以時域信號向量即為[ 1, 1, -1, 1, 1 ]
                # 在接收端扣除cyclic prefix後時域信號向量變回 [ 1, -1, 1, 1 ]，我們會觀察加上cp跟沒加cp，每個取樣點都耗去相同能量，不合理！
                # 所以我們一定要乘上np.sqrt((Nfft+n_guard)/Nfft)
                for m in range(len(y)):
                    y[m] = y[m] * np.sqrt((Nfft + n_guard) / Nfft)


                ######################################################################################################
                # 以上為傳送端
                #
                # 現在將傳送出去的OFDM symbol加上雜訊
                # Eb代表的是平均每個bit的能量
                Eb = 1/K * (Nfft / Nusc) * ((Nfft + n_guard) / Nfft)
                # 若原本一個symbol 能量Es = 1，現在變成1 * (Nfft / Nusc) 可見(式1) ，現在一個取樣點平均能量為1
                # 若原本一個取樣點平均能量為1，加上cp後變成要乘上((Nfft+n_guard) / Nfft)
                # 因為一個symbol含K個bit (所以Eb = Es / K)，這就是為何要除上K
                No = Eb / snr[i]
                for m in range(Nfft + n_guard):
                    y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
                #
                # 以下為接收端
                ######################################################################################################

                #接下來要對接收向量去除cyclic prefix
                y_new = [0]*Nfft
                n = 0
                for m in range(n_guard,Nfft+n_guard,1):
                    y_new[n] = y[m]
                    n += 1
                y = y_new   #現在y已經去除OFDM的cyclic prefix

                for m in range(len(y)):
                    y[m] = y[m] * np.sqrt(Es) * np.sqrt(Nusc) / Nfft  # 因為前面x向量有乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)，所以現在要變回來
                Y = np.fft.fft(y) #現在將y轉到頻域，變成Y

                #進行detection，並計算錯多少個點
                #因為已知通道的頻率響應為H，所以可以藉由Y[m] / H[m] 來解調
                for m in range(Nfft):
                    if (m>2 and m<29) or (m>34 and m<61):
                        # 用Maximum Likelihood來detect symbol
                        min_distance = 10 ** 9
                        for n in range(len(constellation)):
                            if abs(constellation[n] - Y[m]/H[m]) < min_distance:
                                detection = constellation[n]
                                min_distance = abs(constellation[n] - Y[m]/H[m])

                        if detection != X[m]:  # 如果這個sub channel發生symbol error
                            # 不同的symbol調變，就要用不同的方法從symbol error中找bit error
                            if constellation_name == 'QPSK':#QPSK的symbol發生錯誤時
                                # 要確實的找出QPSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                                if abs(detection.real - X[m].real) == 2:
                                    error += 1
                                if abs(detection.imag - X[m].imag) == 2:
                                    error += 1
                            elif constellation_name == '16-QAM':#16-QAM的symbol發生錯誤時
                                # 要確實的找出16-QAM錯幾個bit，而不是找出錯幾個symbol，來估計BER
                                if abs(detection.real - X[m].real) == 2 or abs(detection.real - X[m].real) == 6:
                                    error += 1
                                elif abs(detection.real - X[m].real) == 4:
                                    error += 2
                                if abs(detection.imag - X[m].imag) == 2 or abs(detection.imag - X[m].imag) == 6:
                                    error += 1
                                elif abs(detection.imag - X[m].imag) == 4:
                                    error += 2
                            elif constellation_name == 'BPSK':
                                error += 1
                    else:
                        continue

        ber[i] = error / (Nusc * N * K)  # 分母乘上K是因為一個symbol含有K個bit

    if k==0 and (constellation_name == 'BPSK' or constellation_name == 'QPSK'):
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-theory for BPSK or QPSK (SISO)')
    elif k==0 and constellation_name == '16-QAM':
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-approximation for 16-QAM (SISO)')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-simulation for {0} (multipath = {1})'.format(constellation_name,L))



plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.legend()
plt.grid(True,which='both')
plt.show()

