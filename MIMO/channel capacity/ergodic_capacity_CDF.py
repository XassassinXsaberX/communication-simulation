import numpy as np
import matplotlib.pyplot as plt

#注意:這裡cdf統計的是中斷機率(outage probability)，也就是通道容量小於某個速率的機率

bps = [0]*40
cdf = [0]*40
N = 5000 #執行N次來找CDF

snr_db = 10
snr = np.power(10, snr_db/10)
for i in range(len(bps)):
    bps[i] = 1/2*i

for k in range(2):
    if k == 0:
        Nt = 2
        Nr = 2
    elif k == 1:
        Nt = 4
        Nr = 4

    H = [[0j]*Nt for m in range(Nr)]
    H = np.matrix(H)

    for i in range(len(bps)):
        fail = 0
        for j in range(N):
            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            # 計算目前channel capacity
            capacity = np.log2( np.linalg.det(np.identity(Nr) + snr/Nt * H * H.getH()).real ) #因為det後的值為複數，所以我們取其實部

            if capacity < bps[i]:
                fail += 1

        cdf[i] = fail / N

    plt.plot(bps,cdf,label='Nt={0} , Nr={1}'.format(Nt,Nr))

plt.legend()
#定義x的座標軸間隔，這樣圖片會比較好看
x = [[0] for i in range(10)]
for i in range(len(x)):
    x[i] = 2*i
plt.xticks(x)
plt.grid(True)
plt.title('Distribution of MIMO channel capacity (SNR={0}dB , unknown CSI)'.format(snr_db))
plt.xlabel('Rate (bps/Hz)')
plt.ylabel('CDF')
plt.show()