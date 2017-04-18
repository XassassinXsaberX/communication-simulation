import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*12
snr = [0]*12
capacity = [0]*12
N = 1000 #執行N次來找channel capacity
Nt = 4
Nr = 4
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(2):
    for i in range(len(snr_db)):
        cap = 0

        H = [[0j]*Nt for m in range(Nr)]
        H = np.matrix(H)

        # 定義一個反映天線之間相關性的correlation matrix R
        R = [[0j]*Nt for m in range(Nr)]
        for m in range(Nr):
            for n in range(Nt):
                if m == n :
                    R[m][n] = 1+0j
                elif n > m :
                    if n-m == 1:
                        R[m][n] = 0.76 * np.exp(0.17j*np.pi)
                    elif n-m == 2:
                        R[m][n] = 0.43 * np.exp(0.35j*np.pi)
                    elif n-m == 3:
                        R[m][n] = 0.25 * np.exp(0.53j*np.pi)
                else:
                    if m-n == 1:
                        R[m][n] = 0.76 * np.exp(-0.17j*np.pi)
                    elif m-n == 2:
                        R[m][n] = 0.43 * np.exp(-0.35j*np.pi)
                    elif m-n == 3:
                        R[m][n] = 0.25 * np.exp(-0.53j*np.pi)
        R = np.matrix(R)

        for j in range(N):
            # 先決定i.i.d.的MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            if k == 0: # i.i.d. channel
            #累積所有目前channel matrix的通道容量
                cap += np.log2( np.linalg.det(np.identity(Nr) + snr[i]/Nt * H * H.getH()).real ) #因為det後的值為複數，所以我們取其實部
            elif k == 1: #correlated channel
                # 無法直接R ** (1/2)
                # 需要用SVD分解將R分解成U , S ,V
                # 在將S中的元素取平方根後，得到新的S_new
                # 所以 R**(1/2) = U *  S_new * V
                u,s,v = np.linalg.svd(R)
                u = np.matrix(u)
                v = np.matrix(v)
                s_matrix = [[0] * v.shape[0] for m in range(u.shape[1])]
                for m in range(len(s)):
                    s_matrix[m][m] = np.sqrt(s[m])

                R_root_square = u * s_matrix * v

                # 假設接收端天線之間的correlation matrix 為 I 單位矩陣(即接收端天線間沒有相關性)
                # 只假設傳送端天線有相關性
                H = H * R_root_square

                # 累積所有目前channel matrix的通道容量
                cap += np.log2(np.linalg.det(np.identity(Nr) + snr[i] / Nt * H * H.getH()).real)  # 因為det後的值為複數，所以我們取其實部

        capacity[i] = cap / N

    if k == 0:
        plt.plot(snr_db,capacity,marker='o',label='i.i.d. Nt = {0} , Nr = {1} channel'.format(Nt,Nr))
    elif k == 1:
        plt.plot(snr_db, capacity, marker='o', label='correlated Nt = {0} , Nr = {1} channel'.format(Nt, Nr))

plt.title('ergodic channel capacity (unknown CSI)')
plt.xlabel('Eb/No , dB')
plt.ylabel('bps/Hz')
plt.legend()
plt.grid(True)
plt.show()



