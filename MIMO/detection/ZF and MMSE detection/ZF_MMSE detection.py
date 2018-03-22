import numpy as np
import matplotlib.pyplot as plt
import math

# MMSE 的公式推導可參考 https://www.youtube.com/watch?v=aQqgMcSviko
# 如果要直接看結論公式可快轉到 30:52

snr_db = [0]*13
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
Nt = 2 #傳送端天線數
Nr = 2 #接收端天線數
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

# 定義BPSK星座點
constellation = [ -1, 1 ]
constellation_name="BPSK"

# 利用constellation_num決定要用哪種星座點
constellation_num = 1
if constellation_num == 1:
    # 定義星座點，QPSK symbol值域為{1+j , 1-j , -1+j , -1-j }
    # 則實部、虛部值域皆為{ -1, 1 }
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    constellation_new = [-1, 1]
    constellation_name = 'QPSK'
elif constellation_num == 2:
    # 定義星座點，16QAM symbol值域為{1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j }
    # 則實部、虛部值域皆為{ -3, -1, 1, 3}
    constellation = [1+1j,1+3j,3+1j,3+3j,-1+1j,-1+3j,-3+1j,-3+3j,-1-1j,-1-3j,-3-1j,-3-3j,1-1j,1-3j,3-1j,3-3j]
    constellation_new = [-3, -1, 1, 3]
    constellation_name = '16QAM'
elif constellation_num == 3:
# 定義64QAM星座點
    constellation_new = [-7, -5, -3, -1, 1, 3, 5, 7]
    constellation_name = '64QAM'
    constellation = []
    for i in range(len(constellation_new)):
        for j in range(len(constellation_new)):
            constellation += [constellation_new[i] + 1j * constellation_new[j]]


normalize = 1            # 決定接收端是否要對雜訊normalize (若為0代表不normalize，若為1代表要normalize)


# 根據不同的調變設定snr 間距
for i in range(len(snr)):
    if Nt == 2:
        if constellation_name == 'QPSK':
            snr_db[i] = 2 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 2.5 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 3 * i
        else:
            snr_db[i] = 2 * i
        if normalize == 1:
            snr_db[i] += 10 * np.log10(Nr)
    elif Nt == 3:
        if constellation_name == 'QPSK':
            snr_db[i] = 1.7 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 2.2 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 2.6 * i
        else:
            snr_db[i] = 2 * i
    elif Nt == 4:
        if constellation_name == 'QPSK':
            snr_db[i] = 1.5 * i
        elif constellation_name == '16QAM':
            snr_db[i] = 1.9 * i
        elif constellation_name == '64QAM':
            snr_db[i] = 2.3 * i
        else:
            snr_db[i] = 2 * i
    else:
        snr_db[i] = 2 * i
    snr[i] = np.power(10, snr_db[i] / 10)

#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = [0]*Nt #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = [0]*Nr      #接收端的向量

for k in range(4):
    for i in range(len(snr)):
        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)    # 平均一個symbol有Es的能量
        Eb = Es / K                         # 平均一個bit有Eb能量
        # 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大

        if normalize == 0:
            No = Eb / snr[i]                      # 決定雜訊No
        else:
            No = Eb / snr[i] * Nr

        if k==0:# MRC(1x2) for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k==1:# SISO for BPSK (theory)
            ber[i] = 1 / 2 * (1 - np.sqrt(snr[i] / (snr[i] + 1)))
            # 亦可用以下公式
            #ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue

        for j in range(N):
            #決定要送哪些symbol
            for m in range(Nt): #傳送端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break

            #先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m,n] = 1/np.sqrt(2)*np.random.randn() + 1j/np.sqrt(2)*np.random.randn()

            #接下來決定接收端收到的向量y (共有Nr的元素)
            for m in range(Nr):
                y[m] = 0
            for m in range(Nr):
                for n in range(Nt):
                    y[m] += H[m,n]*symbol[n]
                y[m] += np.sqrt(No/2)*np.random.randn() + 1j*np.sqrt(No/2)*np.random.randn()

            if k == 2:#執行ZF detection
                #決定ZE 的weight matrix
                W = ((H.getH()*H)**(-1))*H.getH()  #W為 Nt x Nr 矩陣
            elif k == 3:  # 執行MMSE detection
                # 決定MMSE 的weight matrix
                W = Es * (Es * H.getH() * H + No * np.identity(Nt)).I * H.getH()  # W為 Nt x Nr 矩陣

            # receive向量 = W矩陣 * y向量
            receive = [0]*Nt
            for m in range(Nt):
                for n in range(Nr):
                    receive[m] += W[m,n]*y[n]

            for m in range(Nt):
                # 接收端利用Maximum Likelihood來detect symbol
                min_distance = 10 ** 9
                for n in range(len(constellation)):
                    if abs(constellation[n] - receive[m]) < min_distance:
                        detection = constellation[n]
                        min_distance = abs(constellation[n] - receive[m])
                # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                if symbol[m] != detection:
                    if constellation_name == 'BPSK':
                        error += 1 # error為bit error 次數
                    elif constellation_name == 'QPSK':
                        if abs(detection.real - symbol[m].real) == 2:
                            error += 1
                        if abs(detection.imag - symbol[m].imag) == 2:
                            error += 1
                    elif constellation_name == '16QAM':
                        if abs(detection.real - symbol[m].real) == 2 or abs(detection.real - symbol[m].real) == 6:
                            error += 1
                        elif abs(detection.real - symbol[m].real) == 4:
                            error += 2
                        if abs(detection.imag - symbol[m].imag) == 2 or abs(detection.imag - symbol[m].imag) == 6:
                            error += 1
                        elif abs(detection.imag - symbol[m].imag) == 4:
                            error += 2
                    elif constellation_name == '64QAM':
                        if abs(detection.real - symbol[m].real) == 2 or abs(detection.real - symbol[m].real) == 6 or abs(detection.real - symbol[m].real) == 14:
                            error += 1
                        elif abs(detection.real - symbol[m].real) == 4 or abs(detection.real - symbol[m].real) == 8 or abs(detection.real - symbol[m].real) == 12:
                            error += 2
                        elif abs(detection.real - symbol[m].real) == 10:
                            error += 3
                        if abs(detection.imag - symbol[m].imag) == 2 or abs(detection.imag - symbol[m].imag) == 6 or abs(detection.imag - symbol[m].imag) == 14:
                            error += 1
                        elif abs(detection.imag - symbol[m].imag) == 4 or abs(detection.imag - symbol[m].imag) == 8 or abs(detection.imag - symbol[m].imag) == 12:
                            error += 2
                        elif abs(detection.imag - symbol[m].imag) == 10:
                            error += 3


        ber[i] = error/(K*Nt*N) #除以K是因為一個symbol有K個bit

    if k==0:
        plt.semilogy(snr_db, ber, marker='o', label='MRC(1x2) for BPSK (theory)')
    elif k==1:
        plt.semilogy(snr_db, ber, marker='o', label='SISO for BPSK (theory)')
    elif k==2:
        plt.semilogy(snr_db, ber, marker='o', label='ZF, Nt={0}, Nr={1}, for {2}'.format(Nt, Nr, constellation_name))
        # 將錯誤率的數據存成檔案
        if N >= 1000000:  # 在很多點模擬分析的情況下，錯誤率較正確，我們可以將數據存起來，之後就不用在花時間去模擬
            with open('ZF detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt, Nr), 'w') as f:
                f.write('snr_db\n')
                for m in range(len(snr_db)):
                    f.write("{0} ".format(snr_db[m]))
                f.write('\nber\n')
                for m in range(len(snr_db)):
                    f.write("{0} ".format(ber[m]))
    elif k==3:
        plt.semilogy(snr_db, ber, marker='o', label='MMSE, Nt={0}, Nr={1}, for {2}'.format(Nt, Nr, constellation_name))
        if N >= 1000000:  # 在很多點模擬分析的情況下，錯誤率較正確，我們可以將數據存起來，之後就不用在花時間去模擬
            with open('MMSE detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt, Nr), 'w') as f:
                f.write('snr_db\n')
                for m in range(len(snr_db)):
                    f.write("{0} ".format(snr_db[m]))
                f.write('\nber\n')
                for m in range(len(snr_db)):
                    f.write("{0} ".format(ber[m]))
plt.legend()
plt.ylabel('ber')
plt.xlabel('snr (Eb/No) dB')
plt.grid(True,which='both')
plt.show()