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


ber1 = [0.22751633333333332 ,0.17589108333333334 ,0.12378241666666667 ,0.075593 ,0.03893641666666667 ,0.017184333333333333 ,0.006974 ,0.0027695833333333335 ,0.0011918333333333334 ,0.0005335833333333333 ,0.00024675 ,0.00011808333333333333 ,5.383333333333333e-05  ]
plt.semilogy(snr,ber1,marker = 'o',label='branch=[2,4,4,8]'.format(constellaion_name))
ber1 = [0.22704633333333332 ,0.1756535 ,0.12408591666666667 ,0.07573733333333334 ,0.038765666666666664 ,0.017073416666666667 ,0.00697075 ,0.0027564166666666666 ,0.0011805 ,0.0005501666666666667 ,0.00024525 ,0.00012225 ,5.6416666666666665e-05  ]
plt.semilogy(snr,ber1,marker = 'o',label='branch=[2,4,6,8]'.format(constellaion_name))
ber1 = [0.22874758333333334 ,0.17685175 ,0.12485525 ,0.07662858333333333 ,0.039952416666666664 ,0.018098916666666666 ,0.007808416666666667 ,0.003387 ,0.0015129166666666667 ,0.0007395833333333333 ,0.0003524166666666667 ,0.00018266666666666667 ,9.258333333333334e-05 ]
plt.semilogy(snr,ber1,marker = 'o',label='branch=[2,2,8,8]'.format(constellaion_name))
ber1 = [0.22716691666666666 ,0.17572925833333333 ,0.1239702 ,0.07576899166666666 ,0.03891104166666667 ,0.017123908333333333 ,0.006928633333333333 ,0.002824333333333333 ,0.0011923583333333333 ,0.0005429666666666666 ,0.0002606416666666667 ,0.00012819166666666666 ,6.3225e-05  ]
plt.semilogy(snr,ber1,marker = 'o',label='branch=[2,4,8,8]'.format(constellaion_name))



if Nr == 2:
    if constellaion_name == 'QPSK':
        ber = [0.08698105, 0.05292785, 0.029031225, 0.01455505, 0.006733625 ,0.002981925 ,0.00126145 ,0.000511875, 0.0002136 ,8.385e-05 ,3.3725e-05, 1.4225e-05, 5.9e-06 ]
    if constellaion_name == '16QAM':
        ber = [0.1686901625, 0.1186008875, 0.0733288, 0.039247575 ,0.018208725, 0.0074320375, 0.0027901125, 0.0009668875, 0.0003223875 ,0.0001040875, 3.46e-05, 1.03375e-05 ,3.15e-06  ]
    if constellaion_name == '64QAM':
        ber = [0.22451341666666666, 0.17355816666666668 ,0.12235333333333333 ,0.07435666666666667, 0.03724375 ,0.015678166666666667, 0.005537916666666667, 0.00178375 ,0.0005044166666666667 ,0.00014083333333333333 ,3.316666666666667e-05 ,8e-06 ,2.1666666666666665e-06 ]
elif Nr == 4:
    if constellaion_name == 'QPSK':
        ber = [0.15562665, 0.118460475, 0.0818836, 0.050323425, 0.02690375, 0.0124315, 0.004925775, 0.00175935, 0.0005531, 0.000163975, 4.89e-05, 1.33e-05, 3.625e-06]
    if constellaion_name == '16QAM':
        ber = [0.243578375, 0.20886575 ,0.1720089375, 0.1315284375, 0.0884094375, 0.0490300625 ,0.021343375, 0.007176 ,0.001872875 ,0.0004275, 8.33125e-05 ,1.625e-05, 3.1375e-06 ]
plt.semilogy(snr,ber,marker = 'o',label='ML detection')
ticks = [0] * 20
for i in range(20):
    ticks[i] = 2 * i
plt.xticks(ticks)
plt.xlim(min(snr) - 1, max(snr) + 1)
plt.xlabel('Eb/No , dB')
plt.ylabel('ber')
plt.grid(True, which='both')
plt.legend()
plt.show()