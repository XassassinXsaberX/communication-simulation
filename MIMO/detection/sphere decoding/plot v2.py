import numpy as np
import matplotlib.pyplot as plt

Nt = 2
Nr = 2

# 利用constellation_num決定要用哪種星座點
constellation_num = 1
if constellation_num == 1:
    constellation_name = 'QPSK'
elif constellation_num == 2:
    constellation_name = '16QAM'
elif constellation_num == 3:
    constellation_name = '64QAM'

# 接下來決定要比較的方法
# 若way == 1 代表只比較所有DFS在不同soft的可能性
# 若way == 2 代表只比較所有Best First Search在不同soft的可能性
# 若way == 3 代表只比較所有BFS在不同soft或不同K的可能性(若way2 = 1代表比較不同soft的結果，若way2 = 2代表比較不同K的結果)
way = 1
way2 = 1
soft = 1
K = 4
if way == 1:
    way_name = 'DFS'
elif way == 2:
    way_name = 'Best First Search'
elif way == 3:
    way_name = 'BFS'
    if way2 == 1:
        way_name2 = 'various soft' # 固定K，改變soft
    elif way2 == 2:
        way_name2 = 'various K'   # 固定soft，改變K

# 先來畫ML detection 的BER
with open('../ML detection/data/ML detection for {0} (Nt={1}, Nr={2}).dat'.format(constellation_name, Nt, Nr)) as f:
    # 以下的步驟都是讀取數據
    f.readline()  # 這一行讀取到字串 "snr_db"
    snr_db_string = f.readline()[:-2]  # 這一行讀取到的是各個 snr 組成的字串
    snr_db = snr_db_string.split(' ')  # 將各snr 組成的字串分開
    for m in range(len(snr_db)):
        snr_db[m] = float(snr_db[m])
    f.readline()  # 這一行讀取到字串 "ber"
    ber_string = f.readline()[:-1]  # 這一行讀取到的是各個 ber 組成的字串
    ber = ber_string.split(' ')  # 將各個ber 組成的字串分開
    for m in range(len(ber)):
        ber[m] = float(ber[m])

    if way == 3:
        if way2 == 1:
            plt.figure('BER({0}), {1}, K={2}'.format(constellation_name, way_name, K1))
        elif way2 == 2:
            plt.figure('BER({0}), {1}, soft={2}'.format(constellation_name, way_name, soft))
    else:
        plt.figure('BER({0}), {1}'.format(constellation_name, way_name))

plt.semilogy(snr_db, ber, marker='o', label='{0} (ML decoding)'.format(constellation_name))

# 再來畫不同方法的錯誤率、複雜度
for i in range(20):
    try:
        if way == 1:
            None
        else:
            with open('./data/soft/{0}/sphere decoding  for {0}, soft = {1}, {2} (Nt=2, Nr=2)'.format(constellation_name, i, way_name)) as f:
                # 以下的步驟都是讀取數據
                f.readline()  # 這一行讀取到字串 "snr_db"
                snr_db_string = f.readline()[:-2]  # 這一行讀取到的是各個 snr 組成的字串
                snr_db = snr_db_string.split(' ')  # 將各snr 組成的字串分開
                for m in range(len(snr_db)):
                    snr_db[m] = float(snr_db[m])

                f.readline()  # 這一行讀取到字串 "ber"
                ber_string = f.readline()[:-2]  # 這一行讀取到的是各個 ber 組成的字串
                ber = ber_string.split(' ')  # 將各個ber 組成的字串分開
                for m in range(len(ber)):
                    ber[m] = float(ber[m])

                f.readline()  # 這一行讀取到字串 "Average visited node"
                node_string = f.readline()[:-2]  # 這一行讀取到的是各個 node數目 組成的字串
                node = node_string.split(' ')  # 將各個node數目 組成的字串分開
                for m in range(len(ber)):
                    node[m] = float(node[m])

                f.readline()  # 這一行讀取到字串 "Addition complexity"
                add_string = f.readline()[:-2]  # 這一行讀取到的是各個 addititon次數 組成的字串
                add = add_string.split(' ')  # 將各個addititon次數 組成的字串分開
                for m in range(len(ber)):
                    add[m] = float(add[m])

                f.readline()  # 這一行讀取到字串 "Multiplication complexity"
                mult_string = f.readline()[:-1]  # 這一行讀取到的是各個 Multiplication次數 組成的字串
                mult = mult_string.split(' ')  # 將各個Multiplication次數 組成的字串分開
                for m in range(len(ber)):
                    mult[m] = float(mult[m])

            plt.figure('BER({0}), soft={1}, {1}'.format(constellation_name, i, way_name))
            plt.semilogy(snr_db, ber, marker='o', label='{0} (ML decoding)'.format(constellation_name))

    except:
        None

plt.figure()