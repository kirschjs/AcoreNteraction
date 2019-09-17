from LECs_interpolation import return_val
import numpy as np
import matplotlib.pyplot as plt

x  = np.arange(0.1,10,0.1)


print("C1_D4")
plt.plot(x,return_val(x,"C1_D4","AB"))
plt.show()
		
print("C1_D3")
plt.plot(x,return_val(x,"C1_D3","ABCD"))
plt.show()		
		
print("C1_D15")
plt.plot(x,return_val(x,"C1_D15","ABC"))
plt.show()		
		
print("nuclear")
plt.plot(x,return_val(x,"nuclear","C"))
plt.show()		
		
print("unitarity")
plt.plot(x,return_val(x,"unitarity","D"))
plt.show()		
		
print("wrong")
try:
	plt.plot(x,return_val(x,"wrong","C"))
	plt.show()
except:
	pass
		
print("wrong again")
try:
	plt.plot(x,return_val(x,"unitarity","wrong"))
	plt.show()
except:
	pass