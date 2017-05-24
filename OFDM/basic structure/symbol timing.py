import numpy as np
import matplotlib.pyplot as plt

# 此模擬的目標是驗證，若送出 { X[0] , X[1] , ... , X[N-1] }，則經過OFDM調變出來的連續的時域信號變為x(t)
# 在不考慮通道及雜訊的情況下，對x(t)取樣後可得 { x[0] , x[1] , ... , x[N-1] }，對其做FFT後是否可得{ X[0] , X[1] , ... , X[N-1] }  (答案是不一定，可能還要乘上Nfft才行！？)
# 要先將取樣後的{ x[0] , x[1] , ... , x[N-1] }全部除上N後在做FFT才能得到{ X[0] , X[1] , ... , X[N-1] }

Nfft = 64                                       # 總共有多少個sub channel
Nusc = 52                                       # 總共有多少sub channel 真正的被用來傳送symbol，假設是sub-channel : 0,1,2,29,30,31及32,33,34,61,62,63不用來傳送symbol
T_symbol = 3.2*10**(-6)                         # ofdm symbol time
t_sample = T_symbol / Nfft                      # 取樣間隔
n_guard = 16                                    # 經過取樣後有n_guard個點屬於guard interval，Nfft個點屬於data interval
X = [0]*Nfft                                    # 從頻域送出64個symbol
x = [0]*Nfft                                    # 在時域取樣後的信號
time = [0]*Nfft                                 # 取樣的時間點

constellation = [-1,1] # 決定星座點

for i in range(Nfft):
    time[i] = i

# 先從頻域決定要送哪些symbol
for i in range(Nfft):
    b = np.random.random()  # 產生一個 (0,1) uniform 分布的隨機變數，來決定要送哪個symbol
    for n in range(len(constellation)):
        if b <= (n + 1) / len(constellation):
            X[i] = constellation[n]
            break

# 送出去的時域信號為x(t) = X[0] * np.exp( j * ( 2 * pi ) * ( 0 / T_symbol ) * t )                          # 載波頻為 0 / T_symbol
#                                            + X[1] * np.exp( j * ( 2 * pi ) * ( 1 / T_symbol ) * t )                          # 載波頻為 1 / T_symbol
#                                            + X[2] * np.exp( j * ( 2 * pi ) * ( 2 / T_symbol ) * t )                          # 載波頻為 2 / T_symbol
#                                            + ...
#                                            + X[Nfft - 1] * np.exp( j * ( 2 * pi ) * ( ( Nfft - 1 ) / T_symbol ) * t )  # 載波頻為 (Nfft - 1) / T_symbol

# 取樣後x[0] = x(0) , x[1] = x(t_sample) , x[2] = x(2*t_sample) .......  x[Nfft -1] = x((Nfft-1)*t_sample)
for i in range(Nfft):
    for j in range(Nfft):
        x[i] += X[j] * np.exp( 1j * (2*np.pi) * ( j/T_symbol ) * ( i*t_sample ) )

# 這是直接對X做IFFT的結果，我們會發現x_ifft 不等於x，而是x_ifft還要在乘上Nfft 才等於x
x_ifft = np.fft.ifft(X)

# 將Nfft個取樣點{ x[0] , x[1] , ... , x[N-1] }，做FFT後要除以Nfft才能得到原本的傳送symbol
X_new = np.fft.fft(x) / Nfft # 如果不除上Nfft會出現嚴重錯誤！？

# 判斷準確的同步取樣後，做fft得到的頻域symbol，與實際的頻域Nfft個子載波送的symbol誤差為多少
error = 0
for i in range(Nfft):
    error += abs(X[i] - X_new[i])**2
error /= Nfft
print(error)

plt.subplot(1,2,1)
plt.title("Actual sampling results (time domain)")
for i in range(Nfft):
    x[i] = abs(x[i])
plt.stem(time,x)
plt.xlabel('sample point n , sample time=(n / T_symbol) , n={0,1,2...Nfft-1}')
plt.ylabel('magnitude')

plt.subplot(1,2,2)
plt.title("direct IFFT results (time domain)")
for i in range(Nfft):
    x_ifft[i] = abs(x_ifft[i])
plt.stem(time,x_ifft)
plt.xlabel('sample point n , sample time=(n / T_symbol) , n={0,1,2...Nfft-1}')
plt.ylabel('magnitude')

plt.show()




