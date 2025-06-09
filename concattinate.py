import pickle
import numpy as np

# Load the pickle files
with open('para1.pkl', 'rb') as f:
    datas1 = pickle.load(f)

with open('para2.pkl', 'rb') as f:
    datas2 = pickle.load(f)

with open('para3.pkl', 'rb') as f:
    datas3 = pickle.load(f)

with open('para4.pkl', 'rb') as f:
    datas4 = pickle.load(f)

# Check they have the same number of samples
assert len(datas1) == len(datas2) == len(datas3) == len(datas4), "Datas must have the same length!"

# Concatenate them
final_list = []

for i in range(len(datas1)):
    a = np.array(datas1[i])
    b = np.array(datas2[i])
    c = np.array(datas3[i])
    d = np.array(datas4[i])

    merged = np.concatenate((a, b, c, d), axis=1)  # Shape: (1, 356)

    final_list.append(merged)

with open('concatenated.pkl', 'wb') as f:
    pickle.dump(final_list, f)
