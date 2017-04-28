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
select_user = 4  # BS會從k_user個使用者中選擇select_user個使用者來送data，注意 Nm * select_user <= Nb，否則會出錯
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [ -1, 1 ]                 #定義星座點的集合
H_DL = [[0j] * Nb for m in range(k_user*Nm)] #先決定k_user個使用者的channel matrix其空間大小為何

for k in range(4):
    for i in range(len(snr_db)):
        error = 0

        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
        # 接下來要算平均一個symbol有多少能量
        energy = 0
        for m in range(len(constellation)):
            energy += abs(constellation[m]) ** 2
        Es = energy / len(constellation)  # 平均一個symbol有Es的能量
        Eb = Es / K                       # 平均一個bit有Eb能量
        # 因為沒有像space-time coding 一樣重複送data，所以Eb不會再變大

        if k == 0:# MRC(1x2) for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k == 1:# SISO for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue


        # 這裡採用 Nb x ( Nm*select_user ) 的MIMO系統，所以通道矩陣為 ( Nm * select_user ) x Nb
        H = [[0j] * Nb for m in range(Nm*select_user)]
        H = np.matrix(H)
        symbol = [0] * (Nm*select_user)         # 雖然BS傳送端有Nb根天線，但實際上一次只會送Nm*select_user個symbol，且Nm*select_user <= Nb
        y = [0] * (Nm*select_user)              # 接收端的向量
        No = Eb * Nb/(Nm*select_user) / snr[i]  # 平均一個symbol會送Nb/(Nm*select_user) * Es，所以平均一個bit會送Nb/(Nm*select_user) * Eb，這是數學推導的結果


        for j in range(N):

            # 接下來我們會從k_user個row vector中 (k_user個使用者的channel vector)
            # 選norm平方較大的select_user個row vector來傳送data
            # ex . 若 select_user = 4，且H_DL 中第3、9、4、10列row vector的norm較大
            #        則BS會選擇送data給第3、9、4、10個使用者
            #

            # 先決定k_user個使用者的channel matrix
            for m in range(k_user*Nm):
                for n in range(Nb):
                    H_DL[m][n] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            # 注意做pre-equalization時，傳送端天線數Nb >= 虛擬接收端天線數( Nm * select_user )
            # 所以我們會從k_user個使用者中，選select_user個擁有較佳通道的使用者來送data

            # 接下來找每個row channel matrix的frobenius norm平方。若norm平方越大，代表BS到該使用者的channel狀況越好
            channel_norm_2 = [[0,0] for m in range(k_user)]# 第一個變數用來存放row matrix的norm平方，第二個變數用來紀錄這是哪一個row matrix(待會排序完後會用到)
            for m in range(k_user):
                channel_norm_2[m][1] = m # 代表這是第m個row matrix(也就是說這個channel matrix為第m個使用者的channel)
                for o in range(Nm):
                    for n in range(Nb):
                        channel_norm_2[m][0] += abs(H_DL[Nm*m+o][n])**2

            # 接下來對norm平方進行排序
            channel_norm_2.sort(reverse=True,key = lambda cust:cust[0])
            # 若沒有排序(即若BS沒有選擇通道條件較好的使用者傳送data時)，BER會變較差

            # 最後可以決定要送data給哪些使用者
            for m in range(select_user):
                for o in range(Nm):
                    for n in range(Nb):
                        H[Nm*m+o,n] = H_DL[Nm*channel_norm_2[m][1]+o][n]


            # 決定要送哪些symbol
            for m in range(len(symbol)):  # 傳送端一次送出select_user個不同symbol
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break

            # 首先要決定weight matrix W
            if k == 2:    # ZF pre-equalization       (channel inversion)
                W = H.getH() * (H*H.getH()).I
            elif k == 3:  # MMSE pre-equalization  (regularized channel inversion)
                W = H.getH() * (H*H.getH() + 1/snr[i]*np.identity(Nm*select_user)).I
            beta = np.sqrt(Nb / complex((W * (W.getH())).trace()))
            W = beta * W

            # 接下來將要送出去symbol vector先乘上W得到codeword向量
            codeword = [0]*Nb
            for m in range(Nb):
                for n in range(Nm*select_user):
                    codeword[m] += W[m,n] * symbol[n]

            # 接下來送出codeword向量，數學模型為 H(matrix)*codeword(vector) + noise(vector)
            # 接下來決定接收端收到的向量y (共有Nm*select_user的元素)
            for m in range(Nm*select_user):
                y[m] = 0
            for m in range(Nm*select_user):
                for n in range(Nb):
                    y[m] += H[m, n] * codeword[n]
                y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

            #接收端收到y向量後先除以beta後，才可以直接解調
            for m in range(Nm*select_user):
                y[m] /= beta

            for m in range(Nm*select_user):
                # 接收端利用Maximum Likelihood來detect symbol
                min_distance = 10 ** 9
                for n in range(len(constellation)):
                    if abs(constellation[n] - y[m]) < min_distance:
                        detection = constellation[n]
                        min_distance = abs(constellation[n] - y[m])
                        # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                if symbol[m] != detection:
                    error += 1  # error為symbol error 次數

        ber[i] = error / (K*Nm*select_user*N)  # 除以K是因為一個symbol有K個bit、分母乘上Nm*select_user是因為傳送端一次送Nm*select_user個元素，而不是Nb個


    if k == 0 :
        plt.semilogy(snr_db,ber,marker='o',label='MRC(1x2) for BPSK (theory)')
    elif k == 1:
        plt.semilogy(snr_db,ber,marker='o',label='SISO for BPSK (theory)')
    elif k == 2:
        plt.semilogy(snr_db,ber,marker='o',label=r'$channel\/inversion:N_b={0},\/N_m={1},\/user:{2}/selected\/user:{3}$'.format(Nb,Nm,k_user,select_user))
    elif k == 3:
        plt.semilogy(snr_db,ber,marker='o',label=r'$regularized\/channel\/inversion:N_b={0}, N_m={1},\/user:{2}/selected\/user:{3}$'.format(Nb,Nm,k_user,select_user))


plt.legend()
plt.ylabel('ber')
plt.xlabel('Eb/No , dB')
plt.grid(True,which='both')
plt.show()
