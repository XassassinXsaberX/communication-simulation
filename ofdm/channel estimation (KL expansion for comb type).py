import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math

# Channel Estimation for OFDM systems (KL expansion)
# pilot arrangement : comb type pilot
# 採用非時變通道
# 可以自由調整 multipath數目L


snr_db = [0]*9
snr = [0]*9
ber = [0]*9
ber_ideal =[0]*9        #用來紀錄，若已知通道情況下，卻還多傳送pilot的BER
mse = [0]*9
N = 100000              #執行N次來找錯誤率
Nfft = 64               #總共有多少個sub channel
Np = 16                 #共16個pilot
X = [0]*64              #從頻域送出64
Nusc = 64               #總共有多少sub channel 真正的被用來傳送symbol，假設所有sub-channel 都有被使用到
pilot_locate = [0]*16   #決定pilt 的位置
for m in range(len(pilot_locate)):
    pilot_locate[m] = m*(Nfft//Np)
t_sample = 5*10**(-8)   #取樣間隔
n_guard = 16            #經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
L = 5                   #假設有L條path  (L越大代表delay越嚴重，freequency selective特性也越嚴重)
h = [0]*L               #每個元素代表L條path各自的impulse response  (因為在一個OFDM symbol period內為常數，所以很明顯是非時變通道)


#當ofdm symbol發生delay一個單位時間，相當於一個往右shift
def shift(x):
    x_new = [0]*len(x)
    for m in range(1,len(x),1):
        x_new[m] = x[m-1]
    return x_new

for i in range(len(snr_db)):
    snr_db[i] = 5*i
    snr[i] = np.power(10,snr_db[i]/10)

plt.figure('1')
plt.figure('2')

# 我們先定義好L個頻域相位偏移向量 v0, v1, v2 ... ,vL
# 分別代表時域向量delay 0 單位時，頻域會出現相位偏移(為有Nfft個點的向量)
# 分別代表時域向量delay 1 單位時，頻域會出現相位偏移(為有Nfft個點的向量)
# ......
# 當時域信號向量delay k單位，頻域信號的向量的元素就要乘上對應的相位偏移
E = [[0]*L for i in range(Nfft)]
for i in range(L):
    for j in range(Nfft):
        E[j][i] = np.exp(-1j*2*np.pi*j*i/Nfft)/np.sqrt(Nfft)

# 決定piltot 位置處的L個頻域相位偏移向量構成的矩陣Ep
Ep = [[0]*len(E[0]) for m in range(Np)]
for m in range(len(E[0])):
    for n in range(Np):
        Ep[n][m] = E[pilot_locate[n]][m]
Ep = np.matrix(Ep)



for k in range(5):
    for i in range(len(snr)):
        MSE = 0
        error = 0
        error_ideal = 0
        if k==0 : # SISO - rayleigh (BPSK) (theory)
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue
        for j in range(N):
            #決定所有sub-channel要送哪些信號
            for m in range(Nfft):
                if (m in pilot_locate) == True: #令pilot一律送出1
                    X[m] = 1
                else:                      #這裡的sub channel 才是送出data
                    b = np.random.random()
                    #採用bpsk調變
                    if b > 0.5:
                        X[m] = 1
                    else:
                        X[m] = -1
            # 將頻域的Nfft個 symbol 做 ifft 轉到時域
            x = np.fft.ifft(X) * Nfft / np.sqrt(Nusc)
            # 乘上Nfft / np.sqrt(Nusc)可將一個OFDM symbol的總能量normalize成 1* Nfft
            # 你可以這樣想，送了 (Nfft - Np) = 48 個symbol，總能量卻為 Nfft = 64，所以平均每symbol花了(64 / 48)的能量-----------------------------------------------------------(式1)
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

            H_real = np.fft.fft(h,Nfft)  # H_real即為multipath channel 的frequency response (有64個點)
            H_real = H_real*np.sqrt((Nfft + n_guard) / Nfft)
            #注意，因為之後y向量的每個元素會乘上np.sqrt((Nfft + n_guard) / Nfft)，所以通道的頻率響應也要跟著乘上np.sqrt((Nfft + n_guard) / Nfft)才行

            # 接下來將每個x的元素乘上np.sqrt((Nfft + n_guard) / Nfft)，代表考慮加上cyclic prefix後多消耗的能量
            for m in range(len(y)):
                y[m] = y[m] * np.sqrt((Nfft + n_guard) / Nfft)


            ######################################################################################################
            # 以上為傳送端
            #
            # 現在將傳送出去的OFDM symbol加上雜訊
            # snr[i] 代表的是平均每個bit的能量
            Eb = 1 * (Nfft / (Nfft-Np)) * ((Nfft + n_guard) / Nfft)
            # 若原本一個symbol 能量Es = 1，現在變成1 * (Nfft / (Nfft-Np))，其中每個取樣點平均能量為1，可見(式1)
            # 若原本一個取樣點平均能量為1，加上cp後變成要乘上((Nfft+n_guard) / Nfft)
            # 因為採用bpsk調變，所以Es = Eb
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
                y[m] = y[m] * np.sqrt(Nusc) / Nfft #因為前面x向量有乘上Nfft / np.sqrt(Nusc)，所以現在要變回來
            Y = np.fft.fft(y) #現在將y轉到頻域，變成Y

            #進行detection，並計算錯多少個點
            #因為接收端不知道通道的頻率響應H，所以要估測 channel

            def interpolate(H,Nfft,Np,method='cubic'):#因為內插法需要Np+1個點，但只有Np個點，所以要特別定義內插函數
                H.append(0)  #要在H中新增一個點，並用線型函數決定這個點的值
                slope = ( H[len(H)-2] - H[len(H)-3] ) / (Nfft/Np)
                H[len(H)-1] = H[len(H)-2] + slope*(Nfft/Np) #決定完成了，可以進行內插
                #現在H有Np+1個點，內插後會變成Nfft+1個點(之後在把最後一個點扣掉，就得到Nfft個點的內差結果)
                x = np.linspace(0, Np, Np+1) # 在 0~Np 中分Np+1個點
                xnew = np.linspace(0, Np, Nfft+1)  # 在 0~Np 中分Nfft+1個點
                f = interp1d(x,H,kind=method)
                H_new = f(xnew)        #內插後會變成Nfft+1個點
                H = [0]*(len(H_new)-1) #要把最後一個點扣掉，變成只有Nfft個點
                for i in range(len(H)):
                    H[i] = H_new[i]
                return H

            if k ==1 :#利用LS來進行channel estimation
                H_LS = [0]*Np
                for m in range(Np):
                    detection = X[pilot_locate[m]]
                    H_LS[m] = Y[pilot_locate[m]] / X[pilot_locate[m]]

                #現在要在H_LS向量中Np個進行內插，內插後變成Nfft個點
                H = interpolate(H_LS,Nfft,Np)
                #H就是估計出來的通道頻率響應，有Nfft個點

            elif k == 2 :#利用KL expansion來進行channel estimation
                H_LS = [0]*Np
                for m in range(Np):
                    detection = X[pilot_locate[m]]
                    H_LS[m] = Y[pilot_locate[m]] / X[pilot_locate[m]]

                #利用LS solution找出c向量
                c = (Ep.getH()*Ep).I*Ep.getH()*(np.matrix(H_LS).transpose())
                #找出c之後，就可以估計channel
                H = np.matrix(E)*c
                H = np.array(H)


                # 以下會畫圖來比較進行內插前後的結果
                #H1 = [0]*Nfft
                #for m in range(Np):
                #    H1[pilot_locate[m]] = Y[pilot_locate[m]] / X[pilot_locate[m]]
                #x1 = [0]*Nfft
                #for m in range(len(x1)):
                #    x1[m] = m
                #plt.plot(x1,np.abs(H_real),marker='o',label='real channel frequency response')
                #plt.plot(x1,np.abs(H), marker='o',label='KL_expansion channel estimation')
                #plt.plot(x1, np.abs(H1),linestyle=' ', marker='o', label='before interpolate')
                #plt.legend()
                #plt.show()

            #估計出通道後就可以來進行detection
            for m in range(Nfft):
                # 求MSE時可將估計出來的通道頻率響應和H_real比較
                MSE += abs(H_real[m] - H[m])**2
                if m not in pilot_locate:
                    # 統計使用estimate的通道來detect，會錯幾個symbol
                    detection = Y[m] / H[m]
                    if abs(detection-1) < abs(detection+1):
                        detection = 1
                    else:
                        detection = -1
                    if detection != X[m]:
                        error += 1

                    # 統計使用真實的通道來detect，會錯幾個symbol
                    detection = Y[m] / H_real[m]
                    if abs(detection - 1) < abs(detection + 1):
                        detection = 1
                    else:
                        detection = -1
                    if detection != X[m]:
                        error_ideal += 1

        ber[i] = error / ((Nfft - Np) * N)              # 使用估測通道後的錯誤率
        ber_ideal[i] = error_ideal / ((Nfft - Np) * N)  # 在已知通道卻仍傳送pilot的最低錯誤率(錯誤率一定最低，因為通道變為已知)
        mse[i] = MSE / (Nfft*N)
    if k==0 :
        plt.figure('1')
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-theory')
    elif k==1:
        plt.figure('1')
        plt.semilogy(snr_db,ber,marker='o',label='LS channel estimation(cubic interpolation)')
        plt.semilogy(snr_db,ber_ideal,marker='o',label='known channel (still transmit pilot)')
        plt.figure('2')
        plt.semilogy(snr_db,mse,marker='o',label='MSE for LS channel estimation(cubic interpolation)')
    elif k==2:
        plt.figure('1')
        plt.semilogy(snr_db,ber,marker='o',label='KL_expansion channel estimation')
        plt.figure('2')
        plt.semilogy(snr_db,mse,marker='o',label='MSE for KL_expansion channel estimation')

plt.figure('1')
plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.title('number of multipath = {0}'.format(L))
plt.legend()
plt.grid(True,which='both')

plt.figure('2')
plt.xlabel('Eb/No , dB')
plt.ylabel('MSE')
plt.title('number of multipath = {0}'.format(L))
plt.grid(True,which='both')
plt.legend()
plt.show()

