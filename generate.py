import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
import os
import time

mpl.use('TkAgg')


class Generator:
    def __init__(self):
        dataframe_sorb = pd.read_excel('Silica-loc-isoth1.xlsx', header=None, sheet_name="Adsorption")
        self.data_sorb = dataframe_sorb.to_numpy()
        dataframe_desorb = pd.read_excel('Silica-loc-isoth1.xlsx', header=None, sheet_name="Desorption")
        self.data_desorb = dataframe_desorb.to_numpy()

        self.pressure_start_index = 21
        self.pore_sizes_s = self.data_sorb[0][1:]
        self.pressures_s = self.data_sorb[:, 0][self.pressure_start_index:]
        self.pore_sizes_d = self.data_desorb[0][1:]
        self.pressures_d = self.data_desorb[:, 0][self.pressure_start_index:]

        self.N = self.data_sorb.shape[0]
        self.P_N = len(self.pressures_s)
        self.n_s = np.zeros(len(self.pressures_s))
        self.n_d = np.zeros(len(self.pressures_d))

        self.picture = None
        self.pore_distribution = None

    def generate_pore_distribution(self, sigma1, sigma2, d0_1, d0_2):
        self.pore_distribution = (1 / sigma1) * np.exp(- np.power((self.pore_sizes_d - d0_1), 2) / (2 * sigma1 ** 2))
        self.pore_distribution += (1 / sigma2) * np.exp(- np.power((self.pore_sizes_d - d0_2), 2) / (2 * sigma2 ** 2))

    def calculate_isotherms(self):
        for p_i in range(self.pressure_start_index, self.data_sorb.shape[0]):
            for d_i in range(len(self.pore_distribution)):
                self.n_s[p_i - self.pressure_start_index] += self.pore_distribution[d_i] * self.data_sorb[p_i][d_i]

        for p_i in range(self.pressure_start_index, self.data_sorb.shape[0]):
            for d_i in range(len(self.pore_distribution)):
                self.n_d[p_i - self.pressure_start_index] += self.pore_distribution[d_i] * self.data_desorb[p_i][d_i]

    def smooth_out_points(self):
        for i in range(1, self.P_N - 1):
            self.n_s[i] = (self.n_s[i - 1] + self.n_s[i] + self.n_s[i + 1]) / 3
            self.n_d[i] = (self.n_d[i - 1] + self.n_d[i] + self.n_d[i + 1]) / 3

    def normalize_data(self):
        max_value = max(self.n_s.max(), self.n_d.max())
        self.n_s = self.n_s / max_value
        self.n_d = self.n_d / max_value

    def plot_data(self):
        plt.plot(self.pressures_d, self.n_d, marker=".")
        plt.plot(self.pressures_d, self.n_s, marker=".")
        plt.show()

    def plot_distribution(self):
        plt.plot(self.pore_sizes_d, self.pore_distribution, marker=".")
        plt.show()

    def generate_picture(self, resolution):
        self.picture = np.zeros((resolution, resolution), dtype=np.bool_)
        utils.graph_to_picture(self.n_s, self.pressures_s, resolution, self.picture)
        utils.graph_to_picture(self.n_d, self.pressures_d, resolution, self.picture)

    def plot_picture(self):
        fig = plt.figure
        plt.imshow(self.picture, cmap='gray', origin='lower')
        plt.show()

    def save_picture(self, params, pic_folder, params_folder, ID):
        if not os.path.exists(pic_folder):
            os.makedirs(pic_folder)
        if not os.path.exists(params_folder):
            os.makedirs(params_folder)
        np.save(os.path.join(pic_folder, ID), self.picture)
        np.save(os.path.join(params_folder, ID), params)


def generate_data_set(sub_folder, base_folder="data"):
    gen = Generator()
    d0_1_range = np.linspace(1, 10, 100)
    d0_2_range = np.linspace(25, 60, 100)
    sigma1 = 3
    sigma2 = 3

    pictures = []
    params = []
    for d0_1 in d0_1_range:
        for d0_2 in d0_2_range:
            gen.generate_pore_distribution(sigma1=sigma1, sigma2=sigma2, d0_1=d0_1, d0_2=d0_2)
            gen.calculate_isotherms()
            gen.smooth_out_points()
            gen.normalize_data()
            gen.generate_picture(resolution=100)
            #gen.save_picture([sigma1, sigma2, d0_1, d0_2], f"data/{sub_folder}/pic", f"data/{sub_folder}/params", ID=f"{ID}")
            pictures.append(gen.picture)
            params.append([sigma1, sigma2, d0_1, d0_2])

    path = f"{base_folder}/{sub_folder}"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(f"{path}/{sub_folder}_x", pictures)
    np.save(f"{path}/{sub_folder}_y", params)


if __name__ == "__main__":
    generate_data_set(sub_folder="test")
    #gen = Generator()
    #gen.generate_pore_distribution(sigma1=3, sigma2=3, d0_1=3, d0_2=20)
    #gen.plot_distribution()
    # gen.calculate_isotherms()
    # gen.smooth_out_points()
    # gen.normalize_data()
    # gen.generate_picture(100)
    # gen.save_picture([2, 5, 9, 20], "data/test/pic", "data/test/params", ID="TEST.txt")
