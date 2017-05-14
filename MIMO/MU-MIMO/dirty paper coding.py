import numpy as np
import matplotlib.pyplot as plt
import math

# 模擬MU-MIMO downlink channel
# 所以BS為傳送端(有Nb根天線)，使用者為接收端
# 共k_user個使用者，每個使用者有Nm根天線，所以接收端共有k_user*Nm根虛擬天線

snr_db = [0]*12
snr = [0]*12
ber = [0]*12
Nb = 4           # BS共Nb根天線
Nm = 1           # 每個使用者有幾根天線
k_user = 10      # 共有k_user個使用者
select_user = 4  # BS會從k_user個使用者中選擇select_user個使用者來送data，注意 Nm * select_user = Nb (若Nb > Nm*select_user，則BS只會選Nm*select_user根天線送data)
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = 2*i
    snr[i] = np.power(10,snr_db[i]/10)

constellation = [ -1, 1 ]                    #定義星座點的集合
K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit
# 接下來要算平均一個symbol有多少能量
energy = 0
for m in range(len(constellation)):
    energy += abs(constellation[m]) ** 2
Es = energy / len(constellation)  # 平均一個symbol有Es的能量
A = 2                             # Tomlinson - Harashima precoding的modulo運算會用到


H_DL = [[0j] * Nb for m in range(k_user*Nm)] #先決定k_user個使用者的channel matrix其空間大小為何

for k in range(4):
    for i in range(len(snr)):
        error = 0                             # 統計bit error數目
        K = int(np.log2(len(constellation)))  # 代表一個symbol含有K個bit

        if k == 0:  # MRC(1x2) for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            ber[i] = ber[i] * ber[i] * (1 + 2 * (1 - ber[i]))
            continue
        elif k == 1:  # SISO for BPSK (theory)
            ber[i] = 1 / 2 - 1 / 2 * np.power(1 + 1 / snr[i], -1 / 2)
            continue

        # 這裡採用 Nb x ( Nm*select_user ) 的MIMO系統，所以通道矩陣為 ( Nm * select_user ) x Nb
        H = [[0j] * Nb for m in range(Nm * select_user)]
        H = np.matrix(H)
        symbol = [0] * (Nm * select_user)  # 雖然BS傳送端有Nb根天線，但實際上一次只會送Nm*select_user個symbol，且Nm*select_user <= Nb
        coding = [0] * (Nm * select_user)  # 將symbol vector進行dirty paper coding後的結果
        y = [0] * (Nm * select_user)  # 接收端的向量

        for j in range(N):
            # 先決定k_user個使用者的channel matrix : H_DL
            for m in range(Nb):
                for n in range(k_user*Nm):
                    H_DL[n][m] = 1 / np.sqrt(2) * np.random.randn() + 1j / np.sqrt(2) * np.random.randn()

            # 接下來BS要選select_user個使用者，來送data
            # 利用遞迴來決定哪些user的通道在dirty paper coding的情況下會比較好
            def recursion(current, select, select_user, k_user, Nb, Nm, H_DL, temp_H, optimal_user, optimal_H, min_max_value):
                # current紀錄目前遞迴到哪個使用者、select為一個list，用來紀錄目前選了哪些user、select_user代表BS要選擇幾個user
                # 總共有k_user個使用者、Nb代表BS天線數、Nm代表user天線數
                # H_DL為BS到k_user個使用者的channel matrix、temp_H是記錄如果選擇某些user而構成的channel matrix
                # optimal_user這個list紀錄要選擇哪些user、optimal_H則記錄最佳使用者所構成的通道為何
                # min_max_value為LQ分解後，L矩陣對角線元素取絕對值後的最小值，在所有可能中的最大值
                if current == select_user:
                    #選擇了select_user個使用者，且每個使用者有Nm根天線
                    m = 0
                    for i in range(len(select)):
                        for j in range(Nm):
                            for k in range(Nb):
                                temp_H[m,k] = H_DL[Nm*select[i]+j][k]
                            m += 1

                    # 接下來對temp_H進行LQ分解
                    Q,R = np.linalg.qr(temp_H.getH())
                    L = R.getH()
                    Q = Q.getH()

                    # 接下來找出temp_H對角線元素取絕對值後的最小值
                    m = 10**9
                    for i in range(min(L.shape[0],L.shape[1])):
                        if abs(L[i,i]) < m:
                            m = abs(L[i,i])
                    if m > min_max_value[0]: # 若找出較佳解，就將矩陣複製過去
                        min_max_value[0] = m
                        for i in range(len(select)):
                            optimal_user[i] = select[i]
                        for i in range(Nm*select_user):
                            for j in range(Nb):
                                optimal_H[i,j] = temp_H[i,j]

                elif current == 0:
                    for i in range(k_user-select_user+1):
                        select[current] = i
                        recursion(current + 1, select, select_user, k_user, Nb, Nm, H_DL, temp_H, optimal_user, optimal_H, min_max_value)
                else:
                    for i in range(select[current-1]+1,k_user-(select_user-current)+1,1):
                        select[current] = i
                        recursion(current + 1, select, select_user, k_user, Nb, Nm, H_DL, temp_H, optimal_user, optimal_H, min_max_value)


            temp_H = np.matrix([[0j]*Nb for m in range(Nm*select_user)])
            select = [0]*select_user
            optimal_user = [0]*select_user
            optimal_H = np.matrix([[0j]*Nb for m in range(Nm*select_user)])
            min_max_value = [0]
            # BS決定要送給哪些使用者資料
            recursion(0, select, select_user, k_user, Nb, Nm, H_DL, temp_H, optimal_user, optimal_H, min_max_value)

            #先對optimal_H進行LQ分解
            Q, R = np.linalg.qr(optimal_H.getH())
            L = R.getH()
            Q = Q.getH()

            # 接下來決定要送哪些symbol
            for m in range(Nm*select_user):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數
                for n in range(len(constellation)):
                    if b <= (n + 1) / len(constellation):
                        symbol[m] = constellation[n]
                        break

            # 定義Tomlinson-Harashiam precoding的modulo運算
            def modulo(x,A):
                real = 2*A*math.floor((x.real + A)/(2*A))
                imag = 2*A*math.floor((x.imag + A)/(2*A))
                return x - (real + 1j*imag)

            # 接下來要對symbol vector進行dirty paper coding
            for m in range(Nm*select_user):
                coding[m] = symbol[m]
                for n in range(m):
                    coding[m] -= coding[n]*L[m,n]/L[m,m]

                if k==3:
                    coding[m] = modulo(coding[m], A) # Tomlinson-Harashiam precoding

            # 最後將剛剛LQ分解得到的Q矩陣取Hermitian變成Q_H再乘上coding vector後得到send vector再送出
            send = [0] * (Nm*select_user)
            Q_H = Q.getH()
            for m in range(len(send)):
                for n in range(Nm*select_user):
                    send[m] += Q_H[m,n] * coding[n]

            # 最後統計一下送了Nm*select_user symbol總共花多少symbol energy
            energy = 0
            for m in range(len(send)):
                energy += abs(send[m])**2
            Es = energy / (Nm*select_user)   # Es代表這次送出的平均symbol energy
            Eb = Es / K                      # Eb代表這次送出的平均bit energy
            No = Eb / snr[i]

            # 接下來將send vector送出
            # 數學模型為 optimal_H(通道矩陣) * send(vector) + noise(vector) = y(receive vector)
            for m in range(Nm*select_user):
                y[m] = 0
            for m in range(Nm*select_user):
                for n in range(Nb):
                    y[m] += optimal_H[m,n] * send[n]
                y[m] += np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()

            # 最後接收端的select_user會各自收到有Nm個元素的向量
            # 除上L矩陣的對角線元素即可進行detection
            for m in range(Nm*select_user):
                y[m] /= L[m,m]
                if k == 3:
                    y[m] = modulo(y[m],A) # 若採用Tomlinson-Harashiam precoding，責接收端的user還需要多做modulo運算

                # 接收端利用Maximum Likelihood來detect symbol
                min_distance = 10 ** 9
                for n in range(len(constellation)):
                    if abs(constellation[n] - y[m]) < min_distance:
                        detection = constellation[n]
                        min_distance = abs(constellation[n] - y[m])
                        # 我們會將傳送端送出的第m個symbol，detect出來，結果為detection

                if symbol[m] != detection:
                    error += 1  # error為symbol error 次數

        ber[i] = error / (K * Nm * select_user * N)  # 除以K是因為一個symbol有K個bit、分母乘上Nm*select_user是因為傳送端一次送Nm*select_user個元素

    if k == 0:
        plt.semilogy(snr_db, ber, marker='o', label='MRC(1x2) for BPSK (theory)')
    elif k == 1:
        plt.semilogy(snr_db, ber, marker='o', label='SISO for BPSK (theory)')
    elif k == 2:
        plt.semilogy(snr_db, ber, marker='o',label=r'$dirty\/paper\/coding(DPC):N_b={0},\/N_m={1},\/user:{2}/selected\/user:{3}$'.format(Nb, Nm,k_user,select_user))
    elif k == 3:
        plt.semilogy(snr_db, ber, marker='o',label=r'$Tomlinson\/Harashima\/DPC:N_b={0}, N_m={1},\/user:{2}/selected\/user:{3}$'.format(Nb, Nm, k_user, select_user))

plt.legend()
plt.ylabel('ber')
plt.xlabel('Eb/No , dB')
plt.grid(True, which='both')
plt.show()







