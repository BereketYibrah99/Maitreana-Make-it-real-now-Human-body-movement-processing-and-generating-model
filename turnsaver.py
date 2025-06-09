import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.optimize import minimize

def load_data(filename):
    with open(filename, 'rb') as f:
        return np.array(pickle.load(f))
data = load_data('vec1.pkl')

par = data.reshape(-1,89)

for j in range(1):
    Set = []
    for i in par:
        x = dst(i, type=2)/ (2 * 89)
        Set.append(x)
    print(j)
    par = Set

with open('para1.pkl', 'wb') as f:
    pickle.dump(par, f)

print("File saved successfully!")

plt.figure(figsize=(10, 6))
for i, sublist in enumerate(par[10:13]):  # Only process sublists from index 30 to 49
    x = range(len(sublist))  # X-axis values (0, 1, 2, ..., len(sublist) - 1)
    y = sublist              # Y-axis values (the data points in the sublist)
    plt.scatter(x, y,s=10)  # s=10 reduces the point size
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plotting All Sublists")
plt.legend()  # Warning: This will add 847 labels to the legend!
plt.grid(True)
plt.show()



