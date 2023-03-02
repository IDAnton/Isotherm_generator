import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

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

    def generate_picture(self, resolution):
        self.picture = np.zeros((resolution, resolution))
        utils.graph_to_picture(self.n_s, self.pressures_s, resolution, self.picture)
        utils.graph_to_picture(self.n_d, self.pressures_d, resolution, self.picture)

    def plot_picture(self):
        fig = plt.figure
        plt.imshow(self.picture, cmap='gray', origin='lower')
        plt.show()


if __name__ == "__main__":
    gen = Generator()
    gen.generate_pore_distribution(sigma1=2, sigma2=5, d0_1=9, d0_2=20)
    gen.calculate_isotherms()
    gen.smooth_out_points()
    gen.normalize_data()
    gen.generate_picture(100)
    gen.plot_picture()
