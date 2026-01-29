import numpy as np
import matplotlib.pyplot as plt

# Array of different input data sizes
N = range(101)

# Amount of computations, classical and quantum
y_knn = N
y_qknn = np.zeros(len(N))
for i in N:
    y_qknn[i] = np.log2(N[i])

# Visualize results
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


