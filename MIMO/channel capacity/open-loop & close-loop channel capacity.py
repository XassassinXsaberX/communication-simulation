import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*11
snr = [0]*len(snr_db)
capacity = [0]*len(snr_db)
Nr = 4
Nt = 4
N = 1000 #執行N次來找channel capacity
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

H = [[0j]*Nt for m in range(Nr)]
H = np.matrix(H)

#water-filling演算法，其流程可參考我的筆記
def water_filling(Nt,snr,r,eigen_value):
    # Nt 為傳送端天線數、snr為訊雜比
    # r 為非零奇異值數目、eigenr_value為一個list，裡面存放非0奇異值
    opt_r = [0]*r
    while True:
        sum_of_inverse_lambda = 0
        length = len(opt_r)
        for i in range(length):
            sum_of_inverse_lambda += 1/eigen_value[i]
        u = Nt/(length) * (1 + 1/snr*sum_of_inverse_lambda)
        for i in range(length):
            opt_r[i] = u- Nt/snr/eigen_value[i]
        if min(opt_r) >= 0:
            return opt_r
        else:
            del opt_r[len(opt_r)-1]


for k in range(2):
    for i in range(len(snr_db)):
        cap = 0
        for j in range(N):
            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            if k == 0:  #接收端未知CSI，所以平均分配能量給每根天線
                # 累積所有目前channel matrix的通道容量
                cap += np.log2(np.linalg.det(np.identity(Nr) + snr[i] / Nt * H * H.getH()).real)  # 因為det後的值為複數，所以我們取其實部

            elif k == 1: #傳送端已知CSI，所以動態分配能量給每根虛擬SISO天線(eigenmode)
                #先找出channel matrix的奇異值
                u,s,v_H = np.linalg.svd(H)
                # u、v_H 為unitary matrix，s為一個list，裡面存放奇異值

                #接下來用 r 來紀錄有多少個非0特徵值，並把它記錄下來
                r = 0
                eigen_value = []
                for m in range(len(s)):
                    if s[m] != 0:
                        r += 1
                        eigen_value.append((s[m]**2)) #H的奇異值平方 = H的特徵值

                # 利用water_filling演算法決定每根虛擬SISO天線(eigenmode)分配到的功率，注意有些虛擬SISO天線(eigenmode)可能分配到功率0
                # 所以opt_r裡面的元素可能會小於Nt，因為有些eigenmode被分配到功率0(即為unused mode)，我就不存到opt_r這個list中
                opt_r = water_filling(Nt,snr[i],r,eigen_value)

                for m in range(len(opt_r)):
                    cap += np.log2(1 + snr[i]/Nt*opt_r[m]*eigen_value[m])

        capacity[i] = cap / N
    if k == 0:
        plt.plot(snr_db, capacity, marker='o', label='channel unknown , Nt = {0} , Nr = {1}'.format(Nt, Nr))
    elif k == 1:
        plt.plot(snr_db, capacity, marker='o', label='channel known , Nt = {0} , Nr = {1}'.format(Nt, Nr))

plt.title('ergodic channel capacity')
plt.xlabel('Eb/No , dB')
plt.ylabel('bps/Hz')
plt.legend()
plt.grid(True)
plt.show()