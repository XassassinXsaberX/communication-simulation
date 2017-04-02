import numpy as np
import matplotlib.pyplot as plt
import math

#reference : http://www.dsplog.com/2008/06/10/ofdm-bpsk-bit-error/

snr_db = [0]*11
snr = [0]*11
ber = [0]*11
N = 10000            #執行N次來找錯誤率
Nfft = 64               #總共有多少個sub channel
X = [0]*64              #從頻域中，64個sub channel對應到的信號
Nusc = 52               #總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
t_sample = 5*10**(-8)   #取樣間隔
n_guard = 16            #經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data

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
                        b = np.random.random()#產生一個 (0,1) uniform 分布的隨機變數
                        #採用bpsk調變
                        if b > 0.5:
                            X[m] = 1
                        else:
                            X[m] = -1
                    else:
                        X[m] = 0
                # 將頻域的Nfft個 symbol 做 ifft 轉到時域
                x = np.fft.ifft(X) * Nfft/np.sqrt(Nusc)
                # 乘上Nfft / np.sqrt(Nusc)可將一個OFDM symbol的總能量normalize成 1* Nfft
                # 你可以這樣想，送了 Nusc = 52 個symbol，總能量卻為 Nfft = 64，所以平均每symbol花了(64 / 52)的能量-----------------------------------------------------------(式1)
                # 而在時域總共有64個取樣點，所以平均每個取樣點的能量為1

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
                # snr[i] 代表的是平均每個bit的能量
                Eb = 1 * (Nfft / Nusc) * ((Nfft+n_guard) / Nfft)
                #若原本一個symbol 能量Es = 1，現在變成1 * (Nfft / Nusc) 可見(式1) ，現在一個取樣點平均能量為1
                #若原本一個取樣點平均能量為1，加上cp後變成要乘上((Nfft+n_guard) / Nfft) 可見(式2)
                # 因為採用bpsk調變，所以Es = Eb
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
                    y[m] = y[m] * np.sqrt(Nusc) / Nfft #因為前面x向量有乘上Nfft/np.sqrt(Nusc)，所以現在要變回來
                Y = np.fft.fft(y) #現在將y轉到頻域，變成Y

                #進行detection，並計算錯多少個點
                for m in range(Nfft):
                    if (m>2 and m<29) or (m>34 and m<61):
                        if abs(Y[m]-1) < abs(Y[m]+1):
                            Y[m] = 1
                        else:
                            Y[m] = -1
                        if Y[m] != X[m]:
                            error += 1
                    else:
                        continue

        ber[i] = error / (Nusc*N)

    if k==0 :
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-theory')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='only AWGN-simulation')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='only AWGN-theory')

plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.legend()
plt.grid(True,which='both')
plt.show()

