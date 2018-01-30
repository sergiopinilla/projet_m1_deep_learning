import numpy as np
import matplotlib.pyplot as plt

mat = np.loadtxt("prec_vs_dset_size.dat", dtype="float")
#print mat
x=mat[:,0]
#print dasi
prec=mat[:,1:-1]
#print prec
y=np.mean(prec,1)
vstd=np.std(prec,1)
#print vstd
plt.figure(figsize=(11.69,8.27))
plt.errorbar(x,y,vstd,fmt='-ko')
#plt.plot(bas, prec, '--ko', markersize=7, label="1000 loops")
plt.grid(True)
plt.xlabel("Dataset size")
plt.ylabel("Precision")
plt.xscale('log')
plt.title('1000 loops, batch size=100, 4 times')
#plt.legend( loc=4)
plt.xlim(70,60000)
#plt.show()
plt.savefig("prec_vs_dset_size.pdf")
