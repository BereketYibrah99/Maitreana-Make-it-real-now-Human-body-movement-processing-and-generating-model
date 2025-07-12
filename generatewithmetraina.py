import numpy as np
import math
import tensorflow as tf
from scipy.fftpack import idst
import matplotlib.pyplot as plt
import pickle

def manual_denormalize(normalized_data):
    # Inverse of manual_normalize
    with open('concatenated.pkl', 'rb') as f:
        data = np.array(pickle.load(f))
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    print(data_min.shape, data_max.shape)
    scale = np.where(data_max != data_min, 2.0 / (data_max - data_min), 1.0)
    original = data_min + (normalized_data + 1.0) / scale
    return original

model = tf.keras.models.load_model('vae_decoder.h5')

latent_dim = model.input_shape[1]
z = tf.random.normal(shape=(600, latent_dim))

# Generate samples from the decoder
generated = manual_denormalize(model.predict(z))
para = []
for i in generated:
    para.append(idst(i, type = 2))

print(len(para))
plt.figure(figsize=(10, 6))
for i, sublist in enumerate(para):  # Only process sublists from index 30 to 49
    x = range(len(sublist))  # X-axis values (0, 1, 2, ..., len(sublist) - 1)
    y = sublist              # Y-axis values (the data points in the sublist)
    plt.scatter(x, y,s = 0.1)  # s=10 reduces the point size
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plotting All Sublists")
plt.legend()  # Warning: This will add 847 labels to the legend!
plt.grid(True)
plt.show()
