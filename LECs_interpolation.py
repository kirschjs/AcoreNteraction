import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def polinv(x, a, b, c,d):
    return a+b/x+c/x**2+d/x**3
def pol(x, a, b, c,d):
    return a+b*x+c*x**2+d*x**3
def pol4(x, a, b, c,d,e):
    return a+b*x+c*x**2+d*x**3+e*x**4



def return_val(lamb,data,var,rang=[0.1,20]):
	
	L = C = D = AB = ABC = ABCD = ABCDE = ABCDEF = []
	
	if (data=="C1_D4"):
			L=[    0.05
		 ,    0.1
		 ,    0.16
		 ,    0.2
		 ,    0.3
		 ,    0.35
		 ,    0.4
		 ,    0.45
		 ,    0.5
		 ,    0.55
		 ,    0.6
		 ,    0.65
		 ,    0.7
		 ,    0.75
		 ,    0.8
		 ,    0.9
		 ,    0.95
		 ,    1
		 ,    1.05
		 ,    1.1
		 ,    1.2
		 ,    1.5
		 ,    2
		 ,    3
		 ,    4
		 ,    6
		 ,    10
		 ]    
			Cc=[	-1.55166500
	,	-2.24166300
	,	-3.44040000
	,	-4.04022700
	,	-6.39651000
	,	-7.80830000
	,	-9.31007578
	,	-10.97613646
	,	-12.78163000
	,	-14.72581116
	,	-16.80907468
	,	-19.03174482
	,	-21.39405000
	,	-23.89403709
	,	-26.53580000
	,	-32.23375000
	,	-35.29122000
	,	-38.48853000
	,	-41.82441000
	,	-45.29949000
	,	-52.66596000
	,	-78.11106000
	,	-131.64598000
	,	-280.45691000
	,	-484.92093000
	,	-1060.79670000
	,	-2880.38650000
	]	
			Dd=[	-0.3242536172
	,	-0.2456829375
	,	0.01905070914
	,	0.223247243
	,	1.206821045
	,	1.980641612
	,	2.82776987
	,	3.913090514
	,	5.208937347
	,	6.717922122
	,	8.468271689
	,	10.46300555
	,	12.73444559
	,	15.27607615
	,	18.13745202
	,	24.81264259
	,	28.66447948
	,	32.91101433
	,	37.52681162
	,	42.57038705
	,	53.97051096
	,	100.476136
	,	232.4126443
	,	854.1074995
	,	2495.364191
	,	16915.63021
	,	756723.2201
	]	
			AB=[	9.68
	,	6.8
	,	5.25
	,	4.85
	,	4.06
	,	3.81
	,	3.64
	,	3.49
	,	3.37
	,	3.27
	,	3.19
	,	3.16
	,	3.05
	,	3
	,	2.96
	,	2.88
	,	2.85
	,	2.82
	,	2.79
	,	2.77
	,	2.72
	,	2.63
	,	2.54
	,	2.45
	,	2.41
	,	2.36
	,	None
	]	
			ABC=[	9.07
	,	6.46
	,	5.09
	,	4.67
	,	3.92
	,	3.69
	,	3.49
	,	3.34
	,	3.22
	,	3.11
	,	3.02
	,	2.95
	,	2.88
	,	2.82
	,	2.77
	,	2.69
	,	2.65
	,	2.62
	,	2.59
	,	2.56
	,	2.51
	,	2.4
	,	2.3
	,	None
	,	None
	,	None
	,	None
	]	
			ABCD=[	8.28
	,	5.97
	,	4.87
	,	4.53
	,	4.02
	,	3.89
	,	3.75
	,	3.64
	,	3.55
	,	3.47
	,	3.39
	,	3.32
	,	3.25
	,	3.19
	,	3.13
	,	3.02
	,	2.97
	,	2.93
	,	2.88
	,	2.84
	,	2.76
	,	2.58
	,	2.37
	,	2.15
	,	2.04
	,	1.94
	,	None
	]	
			ABCDE=[	7.61
	,	5.54
	,	4.67
	,	4.44
	,	4.27
	,	4.27
	,	4.18
	,	4.1
	,	4.02
	,	3.93
	,	3.84
	,	3.75
	,	3.67
	,	3.58
	,	3.5
	,	3.36
	,	3.29
	,	3.23
	,	3.17
	,	3.12
	,	None
	,	2.76
	,	2.49
	,	2.2
	,	2.08
	,	None
	,	None
	]	
			ABCDEF=[	7.06
	,	5.18
	,	4.5
	,	4.4
	,	4.64
	,	4.74
	,	4.66
	,	4.59
	,	4.49
	,	4.37
	,	4.26
	,	4.15
	,	4.04
	,	3.93
	,	3.84
	,	3.66
	,	3.58
	,	3.5
	,	3.43
	,	3.36
	,	3.24
	,	2.95
	,	2.63
	,	None
	,	None
	,	None
	,	None
	]	


			

	elif (data=="C1_D3"):
		L = [	0.05
	,	0.1
	,	0.16
	,	0.2
	,	0.3
	,	0.35
	,	0.4
	,	0.45
	,	0.5
	,	0.55
	,	0.6
	,	0.65
	,	0.7
	,	0.75
	,	0.8
	,	0.9
	,	0.95
	,	1
	,	1.05
	,	1.1
	,	1.2
	,	1.5
	,	2
	,	3
	,	4
	,	6
	,	10
	]	

		Cc = [	-1.55166500
	,	-2.24166300
	,	-3.44040000
	,	-4.04022700
	,	-6.39651000
	,	-7.80830000
	,	-9.31007578
	,	-10.97613646
	,	-12.78163000
	,	-14.72581116
	,	-16.80907468
	,	-19.03174482
	,	-21.39405000
	,	-23.89403709
	,	-26.53580000
	,	-32.23375000
	,	-35.29122000
	,	-38.48853000
	,	-41.82441000
	,	-45.29949000
	,	-52.66596000
	,	-78.11106000
	,	-131.64598000
	,	-280.45691000
	,	-484.92093000
	,	-1060.79670000
	,	-2880.38650000
	]	
		Dd = [	0.1329927285
	,	0.3701120098
	,	1.138524898
	,	1.249910638
	,	2.799024416
	,	3.936630847
	,	5.177438097
	,	6.72964684
	,	8.552092744
	,	10.66423485
	,	13.08996044
	,	15.85408
	,	18.9824788
	,	22.49131818
	,	26.42858035
	,	35.64420878
	,	40.98921369
	,	46.87432648
	,	53.32423158
	,	60.37950916
	,	76.44622944
	,	143.6179272
	,	346.2149284
	,	1468.17912
	,	5254.800421
	,	61617.93639
	,	12838191.7
	]	
		AB = [	9.68
	,	6.8
	,	5.25
	,	4.85
	,	4.06
	,	3.81
	,	3.64
	,	3.49
	,	3.37
	,	3.27
	,	3.19
	,	3.16
	,	3.05
	,	3
	,	2.96
	,	2.88
	,	2.85
	,	2.82
	,	2.79
	,	2.77
	,	2.72
	,	2.63
	,	2.54
	,	2.45
	,	2.41
	,	2.36
	,	None
	]	
		ABC=[	10.33
	,	7.38
	,	6.08
	,	5.36
	,	4.51
	,	4.26
	,	4.04
	,	3.87
	,	3.74
	,	3.62
	,	3.53
	,	3.44
	,	3.37
	,	3.31
	,	3.26
	,	3.17
	,	3.13
	,	3.09
	,	3.06
	,	3.03
	,	2.98
	,	2.87
	,	2.76
	,	2.66
	,	2.60
	,	2.56
	,	None
	]	
		ABCD = [	10.62
	,	7.93
	,	7.33
	,	6.18
	,	5.38
	,	5.12
	,	4.86
	,	4.66
	,	4.49
	,	4.34
	,	4.2
	,	4.08
	,	3.97
	,	3.87
	,	3.79
	,	3.63
	,	3.56
	,	3.5
	,	3.44
	,	3.39
	,	3.29
	,	3.07
	,	2.84
	,	2.6
	,	None
	,	None
	,	None
	]	
		ABCDE = [	10.94
	,	8.78
	,	9.01
	,	7.31
	,	6.36
	,	6.03
	,	5.68
	,	5.41
	,	5.18
	,	4.97
	,	4.78
	,	4.62
	,	4.47
	,	4.34
	,	4.22
	,	4.01
	,	3.92
	,	3.83
	,	3.75
	,	3.68
	,	3.55
	,	3.26
	,	2.95
	,	2.67
	,	None
	,	None
	,	None
	]	
		ABCDEF = [	11.45
	,	10.05
	,	10.66
	,	8.5
	,	7.26
	,	6.83
	,	6.39
	,	6.05
	,	5.75
	,	5.49
	,	5.27
	,	5.06
	,	4.89
	,	4.73
	,	4.58
	,	4.33
	,	4.22
	,	4.12
	,	4.03
	,	3.94
	,	3.79
	,	3.45
	,	3.11
	,	2.88
	,	None
	,	None
	,	None
	]	


	elif (data=="C1_D15"):
		L = [	0.05
	,	0.1
	,	0.16
	,	0.2
	,	0.3
	,	0.35
	,	0.4
	,	0.45
	,	0.5
	,	0.55
	,	0.6
	,	0.65
	,	0.7
	,	0.75
	,	0.8
	,	0.9
	,	0.95
	,	1
	,	1.05
	,	1.1
	,	1.2
	,	1.5
	,	2
	,	3
	,	4
	,	6
	,	10
	]	
		Cc =[	-1.55166500
	,	-2.24166300
	,	-3.44040000
	,	-4.04022700
	,	-6.39651000
	,	-7.80830000
	,	-9.31007578
	,	-10.97613646
	,	-12.78163000
	,	-14.72581116
	,	-16.80907468
	,	-19.03174482
	,	-21.39405000
	,	-23.89403709
	,	-26.53580000
	,	-32.23375000
	,	-35.29122000
	,	-38.48853000
	,	-41.82441000
	,	-45.29949000
	,	-52.66596000
	,	-78.11106000
	,	-131.64598000
	,	-280.45691000
	,	-484.92093000
	,	-1060.79670000
	,	-2880.38650000
	]	
		Dd = [	1.002392192
	,	1.722987346
	,	3.504987582
	,	4.04790139
	,	7.943051578
	,	10.80228028
	,	13.97486057
	,	18.01307982
	,	22.87440111
	,	28.67426894
	,	35.55626025
	,	43.68943032
	,	53.2646948
	,	64.44890335
	,	77.59752149
	,	110.6592829
	,	131.2985101
	,	155.2739701
	,	183.0258812
	,	215.1438822
	,	295.1204267
	,	735.7354446
	,	3201.198262
	,	64382.48444
	,	1658016.226
	,	1972633871
	,	None
	]	
		AB = [	9.68
	,	6.8
	,	5.25
	,	4.85
	,	4.06
	,	3.81
	,	3.64
	,	3.49
	,	3.37
	,	3.27
	,	3.19
	,	3.16
	,	3.05
	,	3
	,	2.96
	,	2.88
	,	2.85
	,	2.82
	,	2.79
	,	2.77
	,	2.72
	,	2.63
	,	2.54
	,	2.45
	,	2.41
	,	2.36
	,	None
	]	
		ABC = [	None
	,	None
	,	None
	,	8.27
	,	7.07
	,	6.76
	,	6.46
	,	6.25
	,	6.09
	,	5.95
	,	5.84
	,	5.74
	,	5.66
	,	5.59
	,	5.53
	,	5.43
	,	5.39
	,	5.35
	,	5.32
	,	5.29
	,	5.23
	,	5.11
	,	5.00
	,	4.88
	,	4.84
	,	None
	,	None
	]	
		ABCD = [	None
	,	None
	,	None
	,	14.95
	,	10.47
	,	9.6
	,	8.71
	,	8.17
	,	7.75
	,	7.41
	,	7.13
	,	6.9
	,	6.71
	,	6.54
	,	6.39
	,	6.15
	,	6.04
	,	5.96
	,	5.88
	,	5.8
	,	5.68
	,	5.4
	,	5.15
	,	None
	,	None
	,	None
	,	None
	]	
		ABCDE = [	None
	,	None
	,	None
	,	14.44
	,	11.16
	,	10.22
	,	9.41
	,	8.83
	,	8.36
	,	7.98
	,	7.65
	,	7.38
	,	7.15
	,	6.94
	,	6.77
	,	6.48
	,	6.36
	,	6.26
	,	6.17
	,	6.09
	,	5.93
	,	5.61
	,	5.37
	,	None
	,	None
	,	None
	,	None
	]	

	elif (data=="nuclear"):
		L=[	0.16
	,	0.2
	,	0.3
	,	0.35
	,	0.4
	,	0.45
	,	0.5
	,	0.55
	,	0.6
	,	0.65
	,	0.7
	,	0.75
	,	0.8
	,	0.85
	,	0.9
	,	0.95
	,	1
	,	1.05
	,	1.1
	,	1.2
	,	1.5
	,	2
	,	3
	,	4
	,	6
	,	10
	]	
		Cc = [	-5.23609711
	,	-6.21155090
	,	-9.04089890
	,	-10.66481819
	,	-12.42824554
	,	-14.33104515
	,	-16.37322998
	,	-18.55475197
	,	-20.87557983
	,	-23.33569527
	,	-25.93508911
	,	-28.67371368
	,	-31.55156555
	,	-34.56868362
	,	-37.72501221
	,	-41.02048416
	,	-44.45520020
	,	-48.02915573
	,	-51.74223328
	,	-59.58599854
	,	-86.45745087
	,	-142.37614441
	,	-295.95935059
	,	-505.20166016
	,	-1090.66113281
	,	-2929.47631836
	]	
		Dd = [	-0.2508527567
	,	-0.01411577035
	,	0.9058334635
	,	1.562725218
	,	2.36543082
	,	3.3244409
	,	4.450396189
	,	5.753778748
	,	7.245120228
	,	8.935015304
	,	10.83416803
	,	12.95315533
	,	15.30282946
	,	17.89442822
	,	20.7388847
	,	23.84721094
	,	27.23128363
	,	30.90286676
	,	34.87341855
	,	43.76101396
	,	78.9169705
	,	172.7038298
	,	559.0134636
	,	1397.569451
	,	6311.303676
	,	89436.18522
	]		

	elif (data=="unitarity"):
		L=[	0.05
	,	0.1
	,	0.16
	,	0.2
	,	0.3
	,	0.35
	,	0.4
	,	0.45
	,	0.5
	,	0.55
	,	0.6
	,	0.65
	,	0.7
	,	0.75
	,	0.8
	,	0.85
	,	0.9
	,	0.95
	,	1.05
	,	1.1
	,	1.2
	,	1.5
	,	2
	,	3
	,	4
	,	6
	,	10
	]
		Cc=[	-0.07525112
	,	-0.20872309
	,	-0.72289535
	,	-1.12549815
	,	-2.47325034
	,	-3.42836099
	,	-4.48771144
	,	-5.66495695
	,	-6.98468235
	,	-8.44796140
	,	-10.05115409
	,	-11.80058087
	,	-13.68878974
	,	-15.69305510
	,	-17.85363976
	,	-20.15208451
	,	-22.58965285
	,	-25.18813501
	,	-30.74383329
	,	-33.78677037
	,	-40.15070955
	,	-62.81802345
	,	-111.61487253
	,	-251.19187583
	,	-446.06292972
	,	-1004.42147643
	,	-2786.87308821
	]	
		Dd=[	-0.6718489361
	,	-1.138950403
	,	-1.440950698
	,	-1.682214684
	,	-2.196869695
	,	-2.163255055
	,	-2.084315563
	,	-1.927048614
	,	-1.617072038
	,	-1.133654812
	,	-0.4706700477
	,	0.4102884562
	,	1.501701299
	,	2.727553437
	,	4.259185312
	,	6.041821303
	,	8.09760687
	,	10.55607482
	,	16.11517017
	,	19.74434205
	,	27.25605369
	,	62.09939789
	,	167.5600148
	,	720.9964392
	,	2255.255789
	,	19497.37226
	,	1333896.806
	]	

	else:
			print(data + " is not a valid set of data.")
			print(" >> you can use C1_D4 C1_D3 C1_D15 nuclear or unitarity")
			return 1
		
		
		
		
		
		
		
	X = np.array(L ).astype(np.double)
	fun = polinv
	if (var == "AB"):
		Y = np.array(AB).astype(np.double)
	elif (var == "ABC"):
		Y = np.array(ABC).astype(np.double)
	elif (var == "ABCD"):
		Y = np.array(ABCD).astype(np.double)
	elif (var == "ABCDE"):
		Y = np.array(ABCDE).astype(np.double)
	elif (var == "ABCDEF"):
		Y = np.array(ABCDEF).astype(np.double)
	elif (var == "C"):
		Y = np.array(Cc).astype(np.double)
		fun = pol
	elif (var == "D"):
		Y = np.array(Dd).astype(np.double)
		fun = pol4
	else:
		print(str(var) + " is not a valid observable or lec.")
		print(" >> You can ask for C D AB ABC ABCD ABCDE or ABCDEF")
		return 1
			
	
	mask       = np.isfinite(Y)
	if len(Y)==0:
		print("Sorry, we dont have the data you are looking for.")
		return 1
	popt, pcov = curve_fit(fun, X[mask],Y[mask] )
	ics        = np.arange(rang[0],rang[1],rang[0])
	if fun == pol4:
		why        = fun(ics, popt[0], popt[1], popt[2], popt[3], popt[4]) 
	else:	
		why        = fun(ics, popt[0], popt[1], popt[2], popt[3]) 
	f2 		   = interp1d(ics,why, kind='cubic')
	#plt.plot(X[mask],Y[mask],'ko',label="data")
	return f2(lamb)
		
		
		
		


		
		
		
		
		