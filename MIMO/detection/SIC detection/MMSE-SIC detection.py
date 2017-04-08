import numpy as np
import matplotlib.pyplot as plt
import math

#reference  http://www.dsplog.com/2008/12/06/mimo-with-mmse-sic-and-optimal-ordering/
snr_db = [0]*12
snr = [0]*12
ber = [0]*12
Nt = 2 #傳送端天線數
Nr = 2 #接收端天線數
N = 10000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)


#這裡採用 Nt x Nr 的MIMO系統，所以通道矩陣為 Nr x Nt
H = [[0j]*Nt for i in range(Nr)]
H = np.matrix(H)
symbol = [0]*Nt #因為有Nt根天線，而且接收端不採用任何分集技術，所以會送Nt個不同symbol
y = [0]*Nr  #接收端的向量

for k in range(6):
    for i in range(len(snr)):
        error = 0
        No = 1/snr[i]  #每個symbol只送一次能量
        # 已知 SNR = Eb / No
        # 令symbol 能量 Es =1 。採用BPSK調變，所以Eb = Es = 1
        # 因為一次只送一個symbol(即沒有重複送相同data)所以Eb = Es = 1
        # 所以 No = 1 / SNR
        if k==2:# MRC(1x2) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k==3:# SISO(BPSK) (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue


        for j in range(N):
            #決定要送哪些symbol (採用BPSK調變)
            for m in range(Nt): #接收端一次送出Nt個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                if b >= 0.5:
                    symbol[m] = 1
                else:
                    symbol[m] = -1

            #先決定MIMO的通道矩陣
            H = [[0j] * Nt for i in range(Nr)]
            H = np.matrix(H)
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

            if k==0:#執行ZF detection
                #決定ZE 的weight matrix
                W = ((H.getH()*H)**(-1))*H.getH()  #W為 Nt x Nr 矩陣
                receive = [0]*Nt
                # W矩陣乘上y向量即可得到估計出來的傳送symbol組成的向量
                for m in range(Nt):
                    for n in range(Nr):
                        receive[m] += W[m,n]*y[n]

                for m in range(Nt):
                    if abs(receive[m]-1) < abs(receive[m]+1):
                        receive_symbol = 1
                    else:
                        receive_symbol = -1
                    if symbol[m] != receive_symbol:
                        error += 1
            elif k==1:#執行MMSE detection
                #決定MMSE 的weight matrix
                W = ((H.getH()*H + 1/snr[i]*np.identity(Nt))**(-1))*H.getH()  #W為 Nt x Nr 矩陣
                receive = [0]*Nt
                #W矩陣乘上y向量即可得到估計出來的傳送symbol組成的向量
                for m in range(Nt):
                    for n in range(Nr):
                        receive[m] += W[m,n]*y[n]

                for m in range(Nt):
                    if abs(receive[m]-1) < abs(receive[m]+1):
                        receive_symbol = 1
                    else:
                        receive_symbol = -1
                    if symbol[m] != receive_symbol:
                        error += 1
            elif k == 4:  # 執行MMSE - SIC detection
                # SIC detection的中心思想為
                # 先利用W的第一列向量和y向量估計出x1，再將y向量減去(h1向量 乘 symbol x1) 得到y1向量，接下來更新W
                # 再利用W的第二列向量和y1向量估計出x2，再將y1向量減去(h1向量 乘 symbol x2) 得到y1向量，接下來更新W
                # ....以此類推
                receive = [0] * Nt
                for m in range(Nt):
                    # 決定MMSE 的weight matrix
                    W = ((H.getH() * H + 1/snr[i]*np.identity((H.getH()*H).shape[0])) ** (-1)) * H.getH()
                    for n in range(Nr):#估計第m個symbol
                        receive[m] += W[0,n] * y[n]
                    #一定要將估計出來的symbol 把他映射到最接近的星座點，否則SIC會失效，其錯誤率變回ZF
                    if abs(receive[m]-1) < abs(receive[m]+1):
                        receive[m] = 1
                    else:
                        receive[m] = -1
                    # 每估計出第1個symbol，就將他乘上對應的通道矩陣的行向量，再被y向量減去，以此更新y向量
                    for n in range(Nr):
                        y[n] -= H[n,0]*receive[m]
                    #接下來要更新通道矩陣H
                    if m == Nt-1:#如果最後一個symbol估計完就不用再更新通道矩陣了
                        break
                    H_new = [[0]*(H.shape[1]-1) for n in range(Nr)]#更新後的通道矩陣會少一行(少第一行)
                    for n in range(Nr):
                        for p in range(len(H_new[0])):
                            H_new[n][p] = H[n,p+1]
                    H = np.matrix(H_new) # H 更新完成

                # 最後算算看解調出的Nt個symbol中，錯了多少個
                for m in range(Nt):
                    if symbol[m] != receive[m]:
                        error += 1
            elif k == 5:  # 執行MMSE - SIC - sort detection
                # SIC-sort detection的中心思想與SIC detection相同
                # 唯一不同的地方在於symbol的檢測順序部再依照原本順序，而是根據通道來決定順序

                #利用每個行向量的norm平方來決定detection順序
                norm = [[0,0] for m in range(Nt)]
                for m in range(Nt):
                    s = 0
                    norm[m][1] = m  #代表第幾個行向量
                    for n in range(Nr):
                        s += H[n,m]*(H[n,m].conjugate())
                    norm[m][0] = s  #代表行向量的norm平方
                norm.sort(key=lambda cost:cost[0],reverse=True)#將通道矩陣H的行向量從大到小順序排列
                #接下來才真正的要將H的每個行向量重新排序
                H_new = [[0]*Nt for m in range(Nr)]
                for m in range(Nt):
                    for n in range(Nr):
                        H_new[n][m] = H[n,norm[m][1]]
                H = np.matrix(H_new)
                # 以上的程式結果為
                # 若H = [ h1   h2   h3   h4 ] 其中h1為行向量
                # 若 | h2 | > | h3 | > | h1 | > | h4 |
                # 則H_new = [ h2   h3   h1   h4 ]


                receive = [0] * Nt
                for m in range(Nt):
                    # 決定MMSE 的weight matrix
                    W = ((H.getH() * H + 1/snr[i]*np.identity((H.getH()*H).shape[0])) ** (-1)) * H.getH()
                    #依序檢測第 norm[m][1] 信號
                    for n in range(Nr):
                        receive[norm[m][1]] += W[0,n]*y[n]
                    # 一定要將估計出來的symbol 把他映射到最接近的星座點，否則SIC會失效，其錯誤率變回ZF
                    if abs(receive[norm[m][1]]-1) < abs(receive[norm[m][1]]+1):
                        receive[norm[m][1]] = 1
                    else:
                        receive[norm[m][1]] = -1
                    # 每估計出第norm[m][1]個symbol，就將他乘上對應的通道矩陣的行向量，再被y向量減去，以此更新y向量
                    for n in range(Nr):
                        y[n] -= receive[norm[m][1]]*H[n,0]

                    # 接下來要更新通道矩陣H
                    if m == Nt - 1:  # 如果最後一個symbol估計完就不用再更新通道矩陣了
                        break
                    H_new = [[0] * (H.shape[1] - 1) for n in range(Nr)]  # 更新後的通道矩陣會少一行(少第一行)
                    for n in range(Nr):
                        for p in range(len(H_new[0])):
                            H_new[n][p] = H[n, p + 1]
                    H = np.matrix(H_new)  # H 更新完成

                #最後算算看解調出的Nt個symbol中，錯了多少個
                for m in range(Nt):
                    if symbol[m] != receive[m]:
                        error += 1


        ber[i] = error/(Nt*N)
    if k==0:
        plt.semilogy(snr_db,ber,marker='o',label='ZF')
    elif k==1:
        plt.semilogy(snr_db,ber,marker='o',label='MMSE')
    elif k==2:
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1X2) (theory)')
    elif k==3:
        plt.semilogy(snr_db,ber,marker='o',label='SISO(BPSK) (theory-formula1)')
    elif k==4:
        plt.semilogy(snr_db, ber, marker='o', label='MMSE-SIC ')
    elif k==5:
        plt.semilogy(snr_db, ber, marker='o', label='MMSE-SIC-sort')
    #elif k==6:
    #    plt.semilogy(snr_db, ber, marker='o', label=' MMSE-SIC-sort')
plt.legend()
plt.ylabel('ber')
plt.xlabel('snr (Eb/No) dB')
plt.grid(True,which='both')
plt.show()

