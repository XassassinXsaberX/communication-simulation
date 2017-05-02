import numpy as np
import matplotlib.pyplot as plt
import math

#reference : http://www.dsplog.com/2008/06/10/ofdm-bpsk-bit-error/

snr_db = [0]*11
snr = [0]*11
ber = [0]*11
N = 100000            #執行N次來找錯誤率
Nfft = 64               #總共有多少個sub channel
X = [0]*64              #從頻域中，64個sub channel對應到的信號
Nusc = 52               #總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
t_sample = 5*10**(-8)   #取樣間隔
n_guard = 16            #經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data

constellation = [1+1j, 1-1j, -1+1j, -1-1j] # 決定QPSK星座點

K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
# 先將所有可能的星座點能量全部加起來
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)      # 從頻域的角度來看，平均一個symbol有Es的能量
Eb = Es / K                           # 從頻域的角度來看，平均一個bit有Eb能量

for i in range(len(snr_db)):
    snr_db[i] = i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(3):
    for i in range(len(snr)):
        error = 0
        if k==0 : # SISO - rayleigh (BPSK) (theory)
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue
        elif k==2: # SISO - only awgn (BPSK) (theory)
            ber[i] = 1/2*math.erfc(np.sqrt(snr[i]))
            continue
        for j in range(N):
            if k==1 :
                #決定所有sub-channel要送哪些信號
                for m in range(64):
                    if (m>2 and m<29) or (m>34 and m<61):#我們規定只有這幾個sub channel能傳送bpsk symbol
                        b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
                        for n in range(len(constellation)):
                            if b <= (n + 1) / len(constellation):
                                X[m] = constellation[n]
                                break
                    else:
                        X[m] = 0

                # 將頻域的Nfft個 symbol 做 ifft 轉到時域
                x = np.fft.ifft(X) * Nfft / np.sqrt(Nusc) / np.sqrt(Es)
                # 乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)可將一個OFDM symbol的總能量normalize成 1* Nfft
                # 你可以這樣想，送了 Nusc = 52 個symbol，總能量卻為 Nfft = 64，所以平均每symbol花了(64 / 52)的能量-----------------------------------------------------------(式1)
                # 而在時域總共有64個取樣點，所以平均每個取樣點的能量為1

                e = 0
                for m in range(64):
                    e += abs(x[m])**2

                # 接下來要加上cyclic prefix
                x_new = [0]*(Nfft+n_guard)
                n = 0
                for m in range(Nfft-n_guard,Nfft,1):
                    x_new[n] = x[m]
                    n += 1
                for m in range(Nfft):
                    x_new[n] = x[m]
                    n += 1
                x = x_new  #現在x已經有加上cyclic prefix

                # 接下來將每個x的元素乘上(Nfft+n_guard)/Nfft，代表考慮加上cyclic prefix後多消耗的能量
                # 如果沒有乘上(Nfft+n_guard)/Nfft ， 那麼有加cyclic prefix 跟沒加cyclic prefix，每個取樣點所耗去的能量都是相同的-->不合理
                # ex
                # 若時域信號向量Nfft = 4個點為 [ 1, -1, 1, 1 ]
                # 加上cyclic prefix後變為Nfft個點為 [ 1, -1, 1, 1 ]   Ng = 1個點，所以時域信號向量即為[ 1, 1, -1, 1, 1 ]
                # 在接收端扣除cyclic prefix後時域信號向量變回 [ 1, -1, 1, 1 ]，我們會觀察加上cp跟沒加cp，每個取樣點都耗去相同能量，不合理！
                # 所以我們一定要乘上np.sqrt((Nfft+n_guard)/Nfft)
                for m in range(len(x)):
                    x[m] = x[m]*np.sqrt((Nfft+n_guard)/Nfft)

                #以下我們可以看到OFDM symbol data 的能量變為80 (energy = 80)
                #若剛剛沒有乘上np.sqrt( (Nfft+n_guard) / Nfft ) 則為64
                #平均每個取樣點的能量從1變成(80 / 64) ------------------------------------------------------------------------------------------(式2)
                #energy = 0
                #for m in range(n_guard,len(x),1):
                #   energy += abs(x[m])*abs(x[m])

                y = x

                ######################################################################################################
                # 以上為傳送端
                #
                # 現在將傳送出去的OFDM symbol加上雜訊
                # Eb代表的是平均每個bit的能量
                Eb = 1/K * (Nfft / Nusc) * ((Nfft+n_guard) / Nfft)
                # 若原本一個symbol 能量Es = 1，現在變成1 * (Nfft / Nusc) 可見(式1) ，現在一個取樣點平均能量為1
                # 若原本一個取樣點平均能量為1，加上cp後變成要乘上((Nfft+n_guard) / Nfft) 可見(式2)
                # 因為一個symbol含K個bit (所以Eb = Es / K)，這就是為何要除上K
                No = Eb / snr[i]
                for m in range(Nfft+n_guard):
                    y[m] +=  np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()
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
                    y[m] = y[m] * np.sqrt(Es) * np.sqrt(Nusc) / Nfft #因為前面x向量有乘上Nfft / np.sqrt(Nusc) / np.sqrt(Es)，所以現在要變回來
                Y = np.fft.fft(y) #現在將y轉到頻域，變成Y

                #進行detection，並計算錯多少個點
                for m in range(Nfft):
                    if (m>2 and m<29) or (m>34 and m<61):
                        # 用Maximum Likelihood來detect symbol
                        min_distance = 10 ** 9
                        for n in range(len(constellation)):
                            if abs(constellation[n] - Y[m]) < min_distance:
                                detection = constellation[n]
                                min_distance = abs(constellation[n] - Y[m])

                        if detection != X[m]: #QPSK的symbol發生錯誤時
                            # 要確實的找出QPSK錯幾個bit，而不是找出錯幾個symbol，來估計BER
                            if abs(detection.real - X[m].real) == 2:
                                error += 1
                            if abs(detection.imag - X[m].imag) == 2:
                                error += 1

                    else:
                        continue

        ber[i] = error / (Nusc*N*K) # 分母乘上K是因為一個symbol含有K個bit

    if k==0 :
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-theory for BPSK')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='only AWGN-simulation for QPSK')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='only AWGN-theory for BPSK')

plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.legend()
plt.grid(True,which='both')
plt.show()

