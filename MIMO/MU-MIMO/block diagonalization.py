import numpy as np
import matplotlib.pyplot as plt

# 模擬MU-MIMO downlink channel
# 所以BS為傳送端(有Nb根天線)，使用者為接收端
# 共k_user個使用者，每個使用者有Nm根天線，所以接收端共有k_user*Nm根虛擬天線

snr_db = [0]*12
snr = [0]*12
ber = [0]*12
Nb = 4           # BS共Nb根天線
Nm = 1           # 每個使用者有幾根天線
k_user = 20      # 共有k_user個使用者
select_user = 4  # BS會從k_user個使用者中選擇select_user個使用者來送data，注意 Nm * select_user  = Nb
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [ -1, 1 ]                 #定義星座點的集合
H_DL = [[0j] * Nb for m in range(k_user*Nm)] #先決定k_user個使用者的channel matrix其空間大小為何

for k in range(4):
    for i in range(len(snr)):
        error = 0

        if k == 0:  # MRC(1x2) for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k == 1:  # SISO for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue

        symbol = []
        for m in range(select_user):
            symbol.append(np.matrix([0j]*Nm).transpose())
        # symbol為一個list，list中的每一個元素為一個(Nm x 1)行向量，代表BS要送給某個user的Nm個symbol

        y = [0] * (Nm * select_user)  # 接收端的向量

        for j in range(N):
            # 接下來我們會從k_user個row vector中 (k_user個使用者的channel vector)
            # 選norm平方較大的select_user個row vector來傳送data
            # ex . 若 select_user = 4，且H_DL 中第3、9、4、10列row vector的norm較大
            #        則BS會選擇送data給第3、9、4、10個使用者
            #

            # 先決定k_user個使用者的channel matrix
            for m in range(k_user * Nm):
                for n in range(Nb):
                    H_DL[m][n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            # 注意做block diagonalization時，傳送端天線數Nb = 虛擬接收端天線數( Nm * select_user )
            # 所以我們會從k_user個使用者中，選select_user個擁有較佳通道的使用者來送data

            # 接下來找每個row channel matrix的frobenius norm平方。若norm平方越大，代表BS到該使用者的channel狀況越好
            channel_norm_2 = [[0, 0] for m in range(k_user)]  # 第一個變數用來存放row matrix的norm平方，第二個變數用來紀錄這是哪一個row matrix(待會排序完後會用到)
            for m in range(k_user):
                channel_norm_2[m][1] = m  # 代表這是第m個row matrix(也就是說這個channel matrix為第m個使用者的channel)
                for o in range(Nm):
                    for n in range(Nb):
                        channel_norm_2[m][0] += abs(H_DL[Nm * m + o][n]) ** 2

            # 接下來對norm平方進行排序
            channel_norm_2.sort(reverse=True, key=lambda cust: cust[0])
            # 若沒有排序(即若BS沒有選擇通道條件較好的使用者傳送data時)，BER會變較差

            # 決定BS到每個user的channel matrix為何
            H_DL_user = []    # 用來存放BS到所有user的channel matrix，為一個list，每個元素為Nm x Nb matrix
            for m in range(select_user):
                # 決定BS到某個user的channel matrix為何
                H_DL_temp = np.matrix([[0j] * Nb for m in range(Nm)])
                for o in range(Nm):
                    for n in range(Nb):
                        H_DL_temp[o,n] = H_DL[Nm * channel_norm_2[m][1] + o][n]
                        #H[Nm * m + o, n] = H_DL[Nm * channel_norm_2[m][1] + o][n]
                H_DL_user.append(H_DL_temp)

            # 接下來決定select_user個user各自的precoding matrix、及channel matrix
            precoding_matrix = []                                     # 用來存放所有選定user的各自的precoding matrix，為一個list，每個元素為Nb x Nm matrix
            H_DL_temp = np.matrix([[0j] * Nb for m in range(Nb-Nm)])   # 要用此matrix來找precoding matrix

            for m in range(select_user):
                # 先來決定H_DL_temp
                r = 0
                for p in range(select_user):
                    if p == m:
                        continue
                    for o in range(Nm):
                        for n in range(Nb):
                            H_DL_temp[r,n] = H_DL_user[p][o,n]
                        r += 1

                # 對某個BS到user的channel matrix做svd分解，得到 U*S*V_H
                # V中的後Nm個行向量所組成的matrix，即為給這個user的precoding matrix
                U, S, V_H = np.linalg.svd(H_DL_temp)
                V = V_H.getH()
                W = np.matrix([[0j] * Nm for m in range(Nb)])  # 為某個user的precoding matrix，為Nb x Nm matrix
                for o in range(Nb):
                    for n in range(Nb-Nm,Nb,1):
                        W[o,n-(Nb-Nm)] = V[o,n]
                precoding_matrix.append(W)

            # 決定BS要送給select_user個使用者哪些symbol (BS一次會送給每個user，Nm個symbol)
            for m in range(select_user):
                for n in range(Nm):
                    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                    for o in range(len(constellation)):
                        if b <= (o + 1) / len(constellation):
                            symbol[m][n,0] = constellation[o]
                            break

            # 接下來將select_user個Nm x 1的symbol vector乘上各自的precoding matrix後，再送出
            transit = np.matrix([0j]*Nb).transpose()
            for m in range(select_user):
                transit += precoding_matrix[m] * symbol[m]

            # 我這裡指是要測試不同的H_DL_user[m] * precoding_matrix[m]是否真的為0
            #for m in range(select_user):
            #    for n in range(select_user):
            #        a = H_DL_user[m] * precoding_matrix[n]


            # 計算一下BS送出去的vector能量總合，以此算出Es、Eb
            K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
            # 接下來要算平均一個symbol有多少能量
            energy = 0
            for m in range(Nb):
                energy += abs(transit[m,0])**2
            Es = energy / (Nm*select_user)    #總共送Nm*select_user個symbol，平均一個symbol有Es能量
            Eb = Es / K                       # 平均一個bit有Eb能量
            No = Eb / snr[i]


            # BS會送出transmit向量
            # 不同的使用者(接收端)，因為BS到該使用者的channel matrix不同，所以transmit會乘上不同的channel matrix
            for m in range(select_user):  #有select_user個使用者
                y = H_DL_user[m] * transit
                for o in range(Nm):
                    y[o,0] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
                # 接下來某個user會收到y向量，假設採用最簡單的ZF解調
                # 決定MMSE detection的weight matrix
                H = H_DL_user[m] * precoding_matrix[m] #對這個user而言，通道向量等同於此向量
                W = (H.getH()*H + 1/snr[i]*np.identity(H.shape[1])).I * H.getH()
                # ZF detection的weight matrix則是
                #W = (H.getH()*H ).I * H.getH()
                # 接收端做equalization : receive向量 = W矩陣 * y向量
                receive = W * y

                for o in range(Nm):
                    # 接收端利用Maximum Likelihood來detect symbol
                    min_distance = 10 ** 9
                    for n in range(len(constellation)):
                        if abs(constellation[n] - receive[o,0]) < min_distance:
                            detection = constellation[n]
                            min_distance = abs(constellation[n] - receive[o,0])
                    # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                    if symbol[m][o,0] != detection:
                        error += 1  # error為symbol error 次數

        ber[i] = error / (K*Nm*select_user*N)  # 除以K是因為一個symbol有K個bit、分母乘上Nm*select_user是因為傳送端一次送Nm*select_user個symbol
    if k == 0 :
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1x2) for BPSK (theory)')
    elif k == 1:
        plt.semilogy(snr_db,ber,marker='o',label='SISO for BPSK (theory)')
    elif k == 2:
        plt.semilogy(snr_db,ber,marker='o',label=r'$block\/diagonalization:N_b={0},\/N_m={1},\/user:{2}/selected\/user:{3}$'.format(Nb,Nm,k_user,select_user))

plt.title('block diagonalization (using MMSE detection in receiver)')
plt.legend()
plt.ylabel('ber')
plt.xlabel('Eb/No , dB')
plt.grid(True,which='both')
plt.show()

