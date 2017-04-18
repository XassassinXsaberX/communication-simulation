import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*12
snr = [0]*12
capacity = [0]*12
N = 5000 #執行N次來找channel capacity
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

for k in range(5):
    for i in range(len(snr_db)):
        cap = 0
        if k == 0:
            Nt = 1
            Nr = 1
        elif k == 1:
            Nt = 1
            Nr = 2
        elif k == 2:
            Nt = 2
            Nr = 1
        elif k == 3:
            Nt = 2
            Nr = 2
        elif k == 4:
            Nt = 4
            Nr = 4

        H = [[0j]*Nt for m in range(Nr)]
        H = np.matrix(H)

        for j in range(N):
            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            #累積所有目前channel matrix的通道容量
            cap += np.log2( np.linalg.det(np.identity(Nr) + snr[i]/Nt * H * H.getH()).real ) #因為det後的值為複數，所以我們取其實部

        capacity[i] = cap / N

    plt.plot(snr_db,capacity,marker='o',label='Nt = {0} , Nr = {1}'.format(Nt,Nr))

plt.title('ergodic channel capacity (unknown CSI)')
plt.xlabel('Eb/No , dB')
plt.ylabel('bps/Hz')
plt.legend()
plt.grid(True)
plt.show()



