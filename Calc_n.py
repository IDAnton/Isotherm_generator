import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 

dataframe1_sorb = pd.read_excel('Silica-loc-isoth1.xlsx', header=None, sheet_name = "Adsorption")
data_sorb = dataframe1_sorb.to_numpy()

dataframe1_desorb = pd.read_excel('Silica-loc-isoth1.xlsx', header=None, sheet_name = "Desorption")
data_desorb = dataframe1_desorb.to_numpy()

n_sorb = np.zeros(458)
n_desorb = np.zeros(458)
d_size = data_sorb[0]
p_distrib = np.zeros(458)
for i in range(458):
    p_distrib[i] = data_sorb[i][0]
d_distrib = np.zeros(len(d_size))

sigma1 = 1
sigma2 = 5
d0_1 = 1
d0_2 = 40

d_distrib = (1/sigma1)* np.exp( - np.power((d_size - d0_1), 2)/(2*sigma1**2))
d_distrib += (1/sigma2) * np.exp( - np.power((d_size - d0_2), 2)/(2*sigma2**2))

for p_i in range(20, 478):
    for d_i in range(len(d_size)):
        n_sorb[p_i-20] += d_distrib[d_i] * data_sorb[p_i][d_i]

for p_i in range(20, 478):
    for d_i in range(len(d_size)):
        n_desorb[p_i-20] += d_distrib[d_i] * data_desorb[p_i][d_i]


figure, axis = plt.subplots(1, 2)        
axis[0].plot(d_size, d_distrib, marker=".")
axis[1].plot(p_distrib, n_sorb, marker=".")
axis[1].plot(p_distrib, n_desorb, marker=".")
plt.show()
