import numpy as np
import matplotlib.pyplot as plt

Nr = 2
constellation_num = 3

if constellation_num == 1:
    constellaion_name = 'QPSK'
elif constellation_num == 2:
    constellaion_name = '16QAM'
elif constellation_num == 3:
    constellaion_name = '64QAM'

if Nr == 2:
    if constellaion_name == 'QPSK':
        snr = [3.010299956639812 ,5.0102999566398125, 7.0102999566398125, 9.010299956639813, 11.010299956639813 ,13.010299956639813, 15.010299956639813, 17.010299956639813, 19.010299956639813 ,21.010299956639813 ,23.010299956639813 ,25.010299956639813 ,27.010299956639813  ]
    if constellaion_name == '16QAM':
        snr = [3.010299956639812, 5.5102999566398125, 8.010299956639813, 10.510299956639813, 13.010299956639813, 15.510299956639813, 18.010299956639813, 20.510299956639813, 23.010299956639813, 25.510299956639813, 28.010299956639813, 30.510299956639813, 33.01029995663981  ]
    if constellaion_name == '64QAM':
        snr = [3.010299956639812, 6.0102999566398125, 9.010299956639813, 12.010299956639813, 15.010299956639813 ,18.010299956639813, 21.010299956639813, 24.010299956639813 ,27.010299956639813 ,30.010299956639813 ,33.01029995663981 ,36.01029995663981 ,39.01029995663981 ]
elif Nr == 4:
    if constellaion_name == 'QPSK':
        snr =  [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0]
    if constellaion_name == '16QAM':
        snr = [0.0 ,1.9 ,3.8 ,5.699999999999999 ,7.6 ,9.5 ,11.399999999999999, 13.299999999999999, 15.2, 17.099999999999998, 19.0, 20.9 ,22.799999999999997 ]
    if constellaion_name == '64QAM':
        snr = [0.0, 2.3, 4.6, 6.8999999999999995, 9.2, 11.5, 13.799999999999999, 16.099999999999998, 18.4, 20.7, 23.0, 25.299999999999997, 27.599999999999998]

visit1 = [32.2726352 ,28.8480256 ,26.4592748 ,24.3636348 ,22.3402476 ,20.6672356 ,19.5080268 ,18.8011916 ,18.4145904 ,18.2129004 ,18.1073984 ,18.0551612 ,18.0271048 ]
plt.plot(snr,visit1,marker = 'o',label='branch=[2,4,4,8]'.format(constellaion_name))
visit1 = [36.8855438 ,32.7874186 ,30.0045016 ,27.5466416 ,25.184768 ,23.188424 ,21.8146656 ,20.9719952 ,20.510232 ,20.256973 ,20.1307022 ,20.0648708 ,20.031732   ]
plt.plot(snr,visit1,marker = 'o',label='branch=[2,4,6,8]'.format(constellaion_name))
visit1 = [34.9454308 ,31.5816074 ,29.2113256 ,27.0095318 ,24.8199536 ,22.9793422 ,21.6896608 ,20.903757 ,20.4692334 ,20.2380074 ,20.1189294 ,20.0588544 ,20.0295078  ]
plt.plot(snr,visit1,marker = 'o',label='branch=[2,2,8,8]'.format(constellaion_name))
visit1 = [41.1228444 ,36.5578374 ,33.4651152 ,30.6679586 ,27.9594814 ,25.6925078 ,24.1043726 ,23.1340458 ,22.58703 ,22.3007818 ,22.1499026 ,22.0745654 ,22.0378546  ]
plt.plot(snr,visit1,marker = 'o',label='branch=[2,4,8,8]'.format(constellaion_name))
#visit1 = [9.934516, 9.444778 ,9.019476, 8.690558 ,8.44767, 8.28379, 8.180962, 8.111144, 8.069182, 8.043392, 8.026854, 8.016508 ,8.010564 ]
#plt.plot(snr,visit1,marker = 'o',label='branch=[2,2,2,2]'.format(constellaion_name))
#visit1 = [50.508786, 43.642692 ,39.078054 ,35.150448, 31.559202 ,28.540386, 26.547534, 25.330914 ,24.683496 ,24.336294 ,24.17574, 24.092952 ,24.042516 ]
#plt.plot(snr,visit1,marker = 'o',label='{0} (sphere decoding) soft=6'.format(constellaion_name))
#visit1 = [59.423658, 50.980706 ,45.671024, 41.06494 ,36.798867, 33.318047 ,30.975903 ,29.562687 ,28.794983, 28.400295, 28.198324 ,28.101647, 28.049497  ]
#plt.plot(snr,visit1,marker = 'o',label='{0} (sphere decoding) soft=7'.format(constellaion_name))
#visit1 = [68.472032 ,58.56952 ,52.337712, 47.032592, 42.155544 ,38.119504, 35.4096 ,33.795168, 32.92536 ,32.456936 ,32.231792 ,32.111984 ,32.060704 ]
#plt.plot(snr,visit1,marker = 'o',label='{0} (sphere decoding) soft=8'.format(constellaion_name))


ticks = [0] * 20
for i in range(20):
    ticks[i] = 2 * i
plt.xticks(ticks)
plt.xlim(min(snr) - 1, max(snr) + 1)

plt.xlabel('Eb/No , dB')
plt.ylabel('Average visited node')
plt.grid(True, which='both')
plt.legend()
plt.show()