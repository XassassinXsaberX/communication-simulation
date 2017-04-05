import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import simulation.multipath
import math

# Channel Estimation for OFDM systems (KL expansion)
# pilot arrangement : block type pilot
# 採用時變通道
# 可以自由調整 multipath數目L、maximum doppler frequency wm


snr_db = [0]*9
snr = [0]*9
ber = [0]*9
ber_ideal = [0]*9
mse = [0]*9
N = 1000000                        #送多少個OFDM symbol來找錯誤率
Nfft = 64                           #總共有多少個sub channel
n_guard = 16                        #經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                        #從頻域送出Nfft個symbol
X_matrix = [[0]*32 for i in range(Nfft)] #從頻域送出Nfft個symbol，我們記錄Nb = 32次
Y_matrix = [[0]*32 for i in range(Nfft)] #從頻域收到Nfft個symbol，我們記錄Nb = 32次
Nusc = 64                           #總共有多少sub channel 真正的被用來傳送symbol，假設所有sub-channel 都有被使用到
N_interval = 4                      #決定每隔多少個OFDM symbol傳送一次pilot
Nb = 32                             #每隔Nb的OFDM symbol就進行一次通道估測
H_matrix = [[0]*Nb for i in range(Nfft)]
                                    #代表經過Nb個OFDM symbol time後估計出來的頻域channel response matrix
                                    #H = [h0 ,h1, h2, ...]
                                    # h0行向量代表，在第0 OFDM symbol time估計的channel vector
                                    # h1行向量代表，在第1 OFDM symbol time估計的channel vector.....
H_real_matrix = [[0]*Nb for i in range(Nfft)]
                                    #代表經過Nb個OFDM symbol time後真實的頻域channel response matrix
                                    #H = [h0 ,h1, h2, ...]
                                    # h0行向量代表，在第0 OFDM symbol time真實的channel vector
                                    # h1行向量代表，在第1 OFDM symbol time真實的channel vector.....
t_sample = 1*10**(-5)               #取樣間隔
Ts = t_sample*(Nfft+n_guard)        #送一個OFDM symbol所需的總時間
L = 1                               #假設有L條path  (L越大代表delay越嚴重，freequency selective特性也越嚴重)
wm = 2*np.pi*1                     #maximum doppler frequency (wm越大，通道時變性越嚴重)
h = [0]*L

count = 0                           #用來紀錄收到幾個OFDM symbol，每收到Nb個OFDM symbol才能進行estimation
total = 0


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

# 我們先定義好三個polynomial多項式向量 p0, p1, p2分別代表常數項，一次項，二次項，每一個多項式向量都有Nb個點
# 其中P = [ p0, p1, p2 ]
P = [[0]*3 for i in range(Nb)]
for i in range(3):
    for j in range(Nb):
        if i==0 :    #決定常數項向量
            P[j][i] = 1/np.sqrt(Nb)
        elif i==1 :   #決定一次項向量
            P[j][i] = (2*j-(Nb-1))/np.sqrt(Nb*(Nb*Nb-1)/3)
        elif i==2 :   #決定二次項向量
            P[j][i] = (6*j*j-6*(Nb-1)*j+(Nb-1)*(Nb-2))/np.sqrt(Nb*(Nb**2-1)*(Nb**2-4)/5)

#決定piltot 位置處的三個多項式向量構成的多項式矩陣Pp
Pp = [[0]*len(P[0]) for m in range(Nb//N_interval)]
for i in range(len(P[0])):
    for j in range(Nb//N_interval):
        Pp[j][i] = P[j*N_interval][i]
Pp = np.matrix(Pp)


print('目前正在處理 N={0}, L={1}, wm={2}'.format(N,L,wm))

for k in range(2):
    for i in range(len(snr)):
        MSE = 0
        MSE_ideal = 0
        error = 0
        error_ideal = 0
        total = 0
        count = 0
        if k==0 : # SISO - rayleigh (BPSK) (theory)
            ber[i] = 1/2*(1-np.sqrt(snr[i]/(snr[i]+1)))
            continue
        for j in range(N):
            #我們要記錄Nb = 32次，每次送出Nfft = 64個sub channel的symbol
            #所以會有32個行向量，每個行向量有64個元素
            #這些行向量組成一個矩陣X

            #決定所有sub-channel要送哪些信號
            if j%N_interval == 0 :#N_interval=4, 所以當在第0,4,8,.....的OFDM symbol，此時所有sub channel都只送pilot
                for m in range(Nfft):
                    X_matrix[m][j%Nb] = 1
                    X[m] = 1
            else:  #此時所有sub channel都送出data
                for m in range(Nfft):
                    b = np.random.random()
                    #採用bpsk調變
                    if b > 0.5:
                        X_matrix[m][j%Nb] = 1
                        X[m] = 1
                    else:
                        X_matrix[m][j%Nb] = -1
                        X[m] = -1

            # 將頻域的Nfft個 symbol 做 ifft 轉到時域
            x = np.fft.ifft(X) * Nfft / np.sqrt(Nusc)
            # 乘上Nfft / np.sqrt(Nusc)可將一個OFDM symbol的總能量normalize成 1* Nfft
            # 你可以這樣想，送了 3*Nfft  = 192 個symbol，總能量卻為 4*Nfft = 256，所以平均每symbol花了(4 / 3)的能量-----------------------------------------------------------(式1)
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

            #接下來產生L條path，且每一條路徑在一個OFDM symbol 周期內不為常數  (所以為時變通道)
            y = [0]*(Nfft+n_guard)  #先將接收端會收到的向量清空
            for m in range(L):
                for n in range(Nfft+n_guard):
                    y[n] += x[n]*1/np.sqrt(L)*simulation.multipath.rayleigh(wm,j*Ts+t_sample*n+m*t_sample,L)[m]
                    #simulation.multipath(wm,j*Ts+t_sample*n+m*t_sample,L)[m]  代表的是實部虛部為獨立且相同的高斯分佈Gaussian( mean = 0, variance = 1/2 )
                    #前面乘上 1/np.sqrt(L) 是為了normalize multipath channel 的平均功率為1
                x = shift(x) # OFDM symbol往右位移一時間單位，相當於做一次shift
            # y[0] ~ y[Nfft+n_guard] 即為經過Rayleigh fading後的OFDM symbol  (尚未加上noise)

            #H_real = np.fft.fft(h,Nfft)  # H_real即為multipath channel 的frequency response (有64個點)
            #H_real = H_real*np.sqrt((Nfft + n_guard) / Nfft)
            #注意，因為之後y向量的每個元素會乘上np.sqrt((Nfft + n_guard) / Nfft)，所以通道的頻率響應也要跟著乘上np.sqrt((Nfft + n_guard) / Nfft)才行

            # 接下來將每個x的元素乘上np.sqrt((Nfft + n_guard) / Nfft)，代表考慮加上cyclic prefix後多消耗的能量
            for m in range(len(y)):
                y[m] = y[m] * np.sqrt((Nfft + n_guard) / Nfft)


            # 讓我們來找在不考慮雜訊下的通道頻率響應為何
            # 先對接收向量去除cyclic prefix
            y_new = [0] * Nfft
            n = 0
            for m in range(n_guard, Nfft + n_guard, 1):
                y_new[n] = y[m]
                n += 1
            # 現在y_new已經去除OFDM的cyclic prefix
            # 接下來將y_new作FFT
            Y = np.fft.fft(y_new) * np.sqrt(Nusc) / Nfft
            # 我們可以找出通道的頻率響應了
            H_real = [0] * Nfft
            for m in range(len(H_real)):
                H_real[m] = Y[m] / X[m]
            # H_real就是我們通道的頻率響應，有 Nfft = 64 個點


            #將此行向量存到 H_real_matrix 的第 j%Nb 個行向量中
            for m in range(Nfft):
                H_real_matrix[m][j%Nb] = H_real[m]

            ######################################################################################################
            # 以上為傳送端
            #
            # 現在將傳送出去的OFDM symbol加上雜訊
            # snr[i] 代表的是平均每個bit的能量
            Eb = 1 * (N_interval / (N_interval-1)) * ((Nfft + n_guard) / Nfft)
            # 若原本一個symbol 能量Es = 1，現在變成1 *(N_interval / (N_interval-1))，其中每個取樣點平均能量為1，可見(式1)
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

            #將其存到Y_matrix中
            for m in range(Nfft):
                Y_matrix[m][j%Nb] = Y[m]


            #進行detection，並計算錯多少個點
            #因為接收端不知道通道的頻率響應H，所以要估測 channel

            if k == 1 :#利用KL expansion來進行channel estimation
                count += 1
                if j%N_interval == 0:#如果這是第0,4,8....個OFDM symbol，則所有sub channel都是送pilot
                    for m in range(Nfft):
                        H_matrix[m][j%Nb] = Y[m] / X[m] #先用LS估計第 N%Nb 個通道行向量的Nfft個元素
                if count == Nb:#如果已經收集到  Nb//N_interval = 32 / 4 = 8 個pilot向量，我們就可以開始估計通道了
                    count = 0
                    #每一次都估計從H_matrix中估計一列channel
                    for m in range(Nfft):
                        H = [0]*Nb                  #對於每個sub channel都會估計出一組channel vector
                        H_LS = [0]*(Nb//N_interval) #將每一列H_matrix中，利用pilot搭配LS算出的元素，存在此向量(有32 / 4 = 8個元素)
                        for n in range(Nb//N_interval):
                            H_LS[n] = H_matrix[m][n*N_interval]

                        # 利用LS solution找出c向量
                        c = (Pp.getH()*Pp).I*Pp.getH()*(np.matrix(H_LS).transpose())
                        # 找出c之後，就可以估計這一列的channel
                        H = np.matrix(P)*c
                        H = np.array(H)
                        #現在成功的估計出一列向量，可以進行一列的detection
                        for n in range(Nb):
                            if n%N_interval != 0:#若該處不是pilot，才要detect
                                MSE += abs(H_real_matrix[m][n] - H[n]) ** 2
                                # 統計使用estimate的通道來detect，會錯幾個symbol
                                detection = Y_matrix[m][n] / H[n]
                                if abs(detection - 1) < abs(detection + 1):
                                    detection = 1
                                else:
                                    detection = -1
                                if detection != X_matrix[m][n]:
                                    error += 1

                                # 統計使用真實的通道來detect，會錯幾個symbol
                                detection = Y_matrix[m][n] / H_real_matrix[m][n]
                                if abs(detection - 1) < abs(detection + 1):
                                    detection = 1
                                else:
                                    detection = -1
                                if detection != X_matrix[m][n]:
                                    error_ideal += 1

                                total += 1   #統計總共送幾個symbol


        ber[i] = error / total               # 使用估測通道後的錯誤率
        ber_ideal[i] = error_ideal / total   # 在已知通道卻仍傳送pilot的最低錯誤率(錯誤率一定最低，因為通道變為已知)
        mse[i] = MSE / total
    if k==0 :
        plt.figure('1')
        plt.semilogy(snr_db,ber,marker='o',label='rayleigh-theory')
    elif k==1:
        plt.figure('1')
        plt.semilogy(snr_db,ber,marker='o',label='KL_expansion channel estimation(block type)')
        plt.semilogy(snr_db,ber_ideal,marker='o',label='known channel')
        plt.figure('2')
        plt.semilogy(snr_db,mse,marker='o',label='MSE for KL_expansion channel estimation(block type)')


plt.figure('1')
plt.xlabel('Eb/No , dB')
plt.ylabel('BER')
plt.title(r'$number\/\/of\/\/multipath={0},\/OFDM\/\/symbol\/\/period={1}(sec),\/ \omega_m={2:0.2f}(rad/s)$'.format(L,Ts,wm))
plt.legend()
plt.grid(True,which='both')

plt.figure('2')
plt.xlabel('Eb/No , dB')
plt.ylabel('MSE')
plt.title(r'$number\/\/of\/\/multipath={0},\/OFDM\/\/symbol\/\/period={1}(sec),\/ \omega_m={2:0.2f}(rad/s)$'.format(L,Ts,wm))
plt.grid(True,which='both')
plt.legend()
print("處理完成")
plt.show()

