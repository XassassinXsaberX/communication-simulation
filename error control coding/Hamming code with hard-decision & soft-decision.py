import numpy as np
import matplotlib.pyplot as plt
import math

# 採用( 7, 4 )Hamming code來編碼   --->   所以一次會將4個bit編碼成7個bit，再送出
# 並應用於SISO傳輸

snr_db = [0]*11
snr = [0]*len(snr_db)
ber = [0]*len(snr_db)
N = 1000000 #執行N次來找錯誤率
for i in range(len(snr)):
    snr_db[i] = i
    snr[i] = np.power(10,snr_db[i]/10)

m_vector = [0]*4
m_vector = np.matrix(m_vector)  # m_vector代表data vector

y = [0]*7   # 接收端會收到的vector

G = [[1, 1, 0, 1, 0, 0, 0],
     [0, 1, 1, 0, 1, 0, 0],
     [1, 1, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 0, 0, 1]]
G = np.matrix(G)  # G為Hamming code的Generator matrix

H = [[1, 0, 0, 1, 0, 1, 1],
     [0, 1, 0, 1, 1, 1, 0],
     [0, 0, 1, 0, 1, 1, 1]]
H = np.matrix(H)  # P為Parity check matrix

codebook = []
# 利用遞迴的方式找出所有可能的m_vector
# 再把m_vector乘上G matrix(加法改成xor)，得到編碼後的向量c(codeword)
# 最後把這個向量存到codebook中
def recursion(current, m_vector, G, codebook):
    if current == m_vector.shape[1]:
        c = m_vector * G
        for i in range(c.shape[1]):
            c[0,i] %= 2
        codebook += list(c)
    else:
        for i in range(2):
            m_vector[0,current] = i
            recursion(current+1, m_vector, G, codebook)
recursion(0, m_vector, G, codebook)  # codebook中的每個元素都是一個編碼後的向量(codeword)(該向量為list物件)，你可以用debug模式來觀察

# 接下來要找出出現不同錯誤時，所對應到的syndrone vector
# 假設該Hamming code只能更正一個bit的錯誤
set_of_error = [0]*len(y)          # 用來紀錄可能發生的錯誤向量(每個向量為list物件)
set_of_syndrone = [0]*len(y)       # 用來紀錄所有可能的錯誤向量對應到的syndrone vector(每個向量為np.matrix物件)
for i in range(len(y)):
    error_vector = [0]*len(y)
    error_vector[i] = 1            # 如果在第i個位置上發生錯誤時
    set_of_error[i] = error_vector
    syndrone = np.matrix(error_vector) * H.T
    for j in range(syndrone.shape[1]):
        syndrone[0,j] %= 2
    set_of_syndrone[i] = syndrone


codebook2 = [0]*len(codebook) # 這本codebook2基本上與codebook相同，只是codebook2將codebook中每個向量(codeword)中的元素0改成 -1
for i in range(len(codebook2)):
    codebook2[i] = np.matrix([0]*codebook[i].shape[1])
    for j in range(codebook2[i].shape[1]):
        codebook2[i][0,j] = codebook[i][0,j]
        if codebook2[i][0,j] == 0:
            codebook2[i][0,j] = -1


for k in range(3):
    for i in range(len(snr)):
        error = 0
        if k == 0: # theory ber for SISO (BPSK)
            ber[i] = 1 / 2 * math.erfc(np.sqrt(snr[i]))
            continue

        for j in range(N):
            # 首先決定要送哪些bit，我們一次會送len(m_vector)個bit的data
            for m in range(m_vector.shape[1]):
                b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個bit
                if b < 0.5:
                    m_vector[0,m] = 0   # 送bit 0
                else:
                    m_vector[0,m] = 1   # 送bit 1

            # 接下來對m_vector進行編碼
            # 把m_vector乘上G matrix(加法改成xor)，得到編碼後的向量c(codeword)
            c = m_vector * G
            for m in range(c.shape[1]):
                c[0,m] = c[0,m] % 2

            # 當c中的元素為0時，我們送-1、當c中的元素為1時，我們送+1
            for m in range(c.shape[1]):
                if c[0,m] == 0:
                    c[0,m] = -1

            # 接下來統計這個要送出去的vector的總能量
            energy = 0
            for m in range(c.shape[1]):
                energy += abs(c[0,m])**2
            Eb = energy / m_vector.shape[1]        # Eb代表平均一個bit的能量
            No = Eb / snr[i]                       # 決定No

            # 接下來考慮雜訊
            for m in range(len(y)):
                y[m] = c[0, m] + np.sqrt(No / 2) * np.random.randn() + 1j * np.sqrt(No / 2) * np.random.randn()
            # y就是接收端會收到的向量

            # 接下來要對接收到的y向量做hard-decision或soft-decision變成r向量(r向量的元素只有0或1)
            r = [0] * len(y)
            if k == 1:  # 採用hard-decision
                for m in range(len(r)):
                    if y[m].real > 0:
                        r[m] = 1
                    else:
                        r[m] = 0

            elif k == 2: # 採用soft-decision
                min_distance = 10 ** 9
                index = 0
                for m in range(len(codebook2)): # codebook2中的每個元素都是一個列向量，代表一個codeword2  (codeword2中的元素其值域為+1、-1)
                    distance = 0
                    # 先算出接收端收到的y向量與codebook2中第m個編碼向量(codebook)的距離(Euclidean distance)
                    for n in range(codebook2[m].shape[1]):
                        distance += abs(y[n] - codebook2[m][0,n])**2
                    distance = np.sqrt(distance)

                    if distance < min_distance:
                        min_distance = distance
                        index = m  # index變數用來紀錄，r可能會指定成哪個codeword

                # 將r向量指定成與y向量最接近的codeword2向量
                # 再將codeword2向量(其值域為+1、-1)轉成對應的codeword向量(其值域為1、0)
                for m in range(len(r)):
                    r[m] = codebook[index][0,m]

            # 現在 r 向量的元素為0、1
            # 接下來將r向量乘上H matrix的轉置(加法改成xor)得到syndrone vector
            syndrone = np.matrix(r) * H.T
            for m in range(syndrone.shape[1]):
                syndrone[0,m] %= 2

            # 接下來尋找syndrone是否存在於set_of_syndrone中
            # 若不存在，代表沒錯誤
            # 若存在，則更正錯誤
            for m in range(len(set_of_syndrone)):
                exist = 1
                index = m
                for n in range(set_of_syndrone[m].shape[1]):
                    if set_of_syndrone[m][0,n] != syndrone[0,n]:
                        exist = 0
                        break
                if exist == 1: # 若找到syndrone存在於set_of_syndrone中
                    break

            if exist == 0: # 如果沒有錯誤就不用更錯
                None
            if exist == 1: # 如果發現有錯誤就要進行更錯
                for m in range(len(r)):
                    r[m] = (r[m] + set_of_error[index][m]) % 2

            # 最後看與m_vector相比錯多少bit
            for m in range(len(r)-m_vector.shape[1], len(r), 1):
                if r[m] != m_vector[0,m-(len(r)-m_vector.shape[1])]:
                    error += 1

        ber[i] = error / (m_vector.shape[1]*N)

    if k == 0:
        plt.semilogy(snr_db, ber, marker='o', label='theory(BPSK) - uncoded')
    elif k == 1:
        plt.semilogy(snr_db, ber, marker='o', label='simulation - Hamming(7,4) (hard)')
    elif k == 2:
        plt.semilogy(snr_db, ber, marker='o', label='simulation - Hamming(7,4) (soft)')

plt.grid(True,which='both')
plt.legend()
plt.xlabel('SNR(Eb/No)  (dB)')
plt.ylabel('ber')
plt.show()

















