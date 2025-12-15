import numpy as np
import matplotlib.pyplot as plt

N = range(101)

y_knn = N
y_qknn = np.zeros(len(N))
for i in N:
    y_qknn[i] = np.log2(N[i])

plt.plot(N,y_knn,label='KNN')
plt.plot(N,y_qknn,label='QKNN')
plt.ylabel('Computations ($O$)')
plt.xlabel('Input data size ($N$)')
plt.grid()
plt.legend()
plt.savefig("Figures/QKNN_complexity.png")
plt.ylim([0,100])
plt.xlim([0,100])
plt.show()


