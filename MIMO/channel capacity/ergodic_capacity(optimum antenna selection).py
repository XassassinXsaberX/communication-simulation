import numpy as np
import matplotlib.pyplot as plt

snr_db = [0]*11
snr = [0]*11
capacity = [0]*11
Nt = 4
Nr = 4
select_antenna = [0]*Nt #裡面是存放要選擇幾根發射天線來使用
# 舉例來說，若select_antenna = [1,2,3,4]
# 則分別代表select_antenna分別會選擇1根或2根或3根或4根發射天線來送資料
for i in range(Nt):
    select_antenna[i] = i+1
N = 1000 #執行N次來找channel capacity
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

H = [[0j]*Nt for m in range(Nr)]
H = np.matrix(H)

for k in range(len(select_antenna)):
    for i in range(len(snr_db)):
        cap = 0

        for j in range(N):
            # 先決定MIMO的通道矩陣
            for m in range(Nr):
                for n in range(Nt):
                    H[m, n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            # 接下來要決定要選擇哪幾根天線來傳送資料
            # 先定義一個function，利用遞迴來找出所有可能組合，並找出其中最大的channel capacity
            def optimal_antenna_selection(H,Q,Nt,current,snr,selection,max_capacity):
                # H為目前的channel matrix、Q為共要選擇幾根天線、Nt為傳送端天線數、current為目前已選擇幾根天線了
                # snr為訊雜比、selection為一個list，裡面存放你選擇哪幾根天線，max_capacity為目前統計最大的通道容量為何
                if current == Q: #若目前已選擇Q根天線
                    H_new = [[0j]*Q for i in range(H.shape[0])] #H_new 為 Nr x Q 的矩陣
                    H_new = np.matrix(H_new)
                    for i in range(H_new.shape[1]):
                        for j in range(H_new.shape[0]):
                            H_new[j,i] = H[j,selection[i]]
                    #找出我們選擇這Q根天線後所形成的新channel matrix : H_new，其channel capacity
                    capacity = np.log2( np.linalg.det(np.identity(Nr) + snr/Q * H_new * H_new.getH()).real ) #因為det後的值為複數，所以我們取其實部
                    if capacity > max_capacity[0]:
                        max_capacity[0] = capacity
                else:
                    if current == 0:#若目前選擇0根天線
                        for i in range(Nt-Q+1): #還需要決定選擇哪Q根天線
                            selection[current] = i
                            optimal_antenna_selection(H, Q, Nt, current+1, snr, selection, max_capacity)
                    else:
                        for i in range(selection[current-1]+1,Nt-(Q-current)+1): #還需要決定選擇哪 ( Q - current ) 根天線
                            selection[current] = i
                            optimal_antenna_selection(H, Q, Nt, current+1, snr, selection, max_capacity)


            max_capacity = [0]    # 待會要用來紀錄，經過antenna selection後最佳的通道容量
            Q = select_antenna[k] # 代表要選擇幾根天線
            selection = [0]*Q     # selection為一個list，裡面存放你選擇哪幾根天線
            optimal_antenna_selection(H, Q, Nt, 0, snr[i], selection, max_capacity)

            #累積所有目前channel matrix的通道容量
            cap += max_capacity[0]

        capacity[i] = cap / N

    plt.plot(snr_db,capacity,marker='o',label='Nt = {0} , Nr = {1} , Q={2}'.format(Nt,Nr,Q))

plt.title('ergodic channel capacity (unknown CSI)  optimal antenna selection')
plt.xlabel('Eb/No , dB')
plt.ylabel('bps/Hz')
plt.legend()
plt.grid(True)
plt.show()



