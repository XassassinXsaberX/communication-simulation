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
            def sub_optimal_antenna_selection(H,Q,Nt,snr,type):
                # H為目前的channel matrix、Q為共要選擇幾根天線、Nt為傳送端天線數
                # snr為訊雜比、type用來代表你是選擇哪種sub-optimal antenna selection

                if type == 0: # ascending order antenna selection
                    selection = []  # 用來紀錄你選擇了哪些天線
                    remain_antenna = [0] * Nt  # 用來存放尚未被選取的天線

                    for i in range(Nt):
                        remain_antenna[i] = i
                    for i in range(Q):#總共要找Q根天線
                        #決定第 i 根天線為何
                        selection.append(0)
                        max_capacity = 0   #用來紀錄如果選了i根天線的sub-optimal channel capacity
                        H_new = [[0j]*(i+1) for m in range(H.shape[0])] #目前共決定( i + 1)根天線
                        H_new = np.matrix(H_new)
                        for j in range(len(remain_antenna)):
                            selection[i] = remain_antenna[j] #假如第i根天線決定選第remain_antenna[j]根天線
                            sort_selection = sorted(selection)  # 將天線編號從小到大排列

                            #不使用排序得到的結果也相同....(不建議這樣用)
                            for m in range(i+1): #目前共決定了( i + 1 )根天線，所以H_new會有( i + 1 )行
                                for n in range(H.shape[0]):
                                    #H_new[n,m] = H[n,selection[m]]
                                    H_new[n,m] = H[n,sort_selection[m]]

                            # 找出我們選擇這( i + 1)根天線後所形成的新channel matrix : H_new，其channel capacity
                            capacity = np.log2( np.linalg.det(np.identity(Nr) + snr / (i+1) * H_new * H_new.getH()).real )  # 因為det後的值為複數，所以我們取其實部
                            if capacity > max_capacity:
                                max_capacity = capacity
                                select_j_antenna = j # 從剩下的remain_antenna中選擇第 j 個天線，把這個index記錄下來，等會要用到

                        selection[i] = remain_antenna[select_j_antenna]  #我們最後決定選擇第remain_antenna[select_j_antenna]根天線
                        del remain_antenna[select_j_antenna]   #因為決定選第remain_antenna[select_j_antenna]根天線，所以會從這個list中去除
                    return max_capacity

                elif type == 1: #descending order antenna selection
                    selection = [0]*Nt  # 用來紀錄你選擇了哪些天線
                    for i in range(Nt):
                        selection[i] = i

                    if Q < Nt:
                        for i in range(Nt - Q):  # 總共要找Q根天線，因為是降序搜尋，所以一開始已經找到Nt根天線，依序一個一個天線扣除，直到剩Q根天線
                            # 決定第 i 次要哪根天線被扣除

                            max_capacity = 0  # 用來紀錄如果選了(Nt - i - 1)根天線的sub-optimal channel capacity
                            H_new = [[0j] * (Nt - i - 1) for m in range(H.shape[0])]  # 再扣除一次天線後，目前會剩下( Nt - i - 1)根天線
                            H_new = np.matrix(H_new)
                            for j in range(len(selection)):
                                now_selection = [0] * (Nt - i )  # 扣除第i次天線前，還有(Nt - i)根天線
                                # 將selection中的元素複製到now_selection
                                for m in range(len(selection)):
                                    now_selection[m] = selection[m]
                                del now_selection[j]  # 扣除所有剩下的天線中，第j根天線

                                # 找出扣除這跟天線後所形成的新的channel matrix
                                for m in range(len(now_selection)):
                                    for n in range(H.shape[0]):
                                        H_new[n, m] = H[n, now_selection[m]]


                                # 找出扣除這根天線後所形成的新channel matrix : H_new，其channel capacity
                                capacity = np.log2( np.linalg.det(np.identity(Nr) + snr / (Nt - i - 1) * H_new * H_new.getH()).real )  # 因為det後的值為複數，所以我們取其實部
                                if capacity > max_capacity:
                                    max_capacity = capacity
                                    select_j_antenna = j  # 從剩下的remain_antenna中選擇第 j 個天線，把這個index記錄下來，等會要用到

                            del selection[select_j_antenna]  # 因為決定選remain_antenna[select_j_antenna]，所以會從這個list中去除
                        return max_capacity

                    else:
                        max_capacity = np.log2( np.linalg.det(np.identity(Nr) + snr / Nt * H * H.getH()).real )  # 因為det後的值為複數，所以我們取其實部
                        return max_capacity



            Q = select_antenna[k] # 代表要選擇幾根天線
            max_capacity = sub_optimal_antenna_selection(H, Q, Nt, snr[i], 1)

            #累積所有目前channel matrix的通道容量
            cap += max_capacity

        capacity[i] = cap / N

    plt.plot(snr_db,capacity,marker='o',label='Nt = {0} , Nr = {1} , Q={2}'.format(Nt,Nr,Q))

plt.title('ergodic channel capacity (unknown CSI)  sub-optimal antenna selection')
plt.xlabel('Eb/No , dB')
plt.ylabel('bps/Hz')
plt.legend()
plt.grid(True)
plt.show()



