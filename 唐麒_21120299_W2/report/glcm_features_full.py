import numpy as np


class glcm_features:
    def __init__(self, glcm, gray_level):
        self.glcm = glcm
        self.gray_level = gray_level

    def calculate_glcm_mean(self):
        mean = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                mean += self.glcm[i, j] * i / self.gray_level ** 2
        return mean

    def calculate_glcm_variance(self):
        mean = self.calculate_glcm_mean()
        variance = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                variance += self.glcm[i, j] * (i - mean) ** 2
        return variance

    def calculate_glcm_inertia(self):
        inertia = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                inertia += 1 / (1 + (i - j) ** 2) * self.glcm[i, j]
        return inertia

    def calculate_glcm_contrast(self):
        contrast = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                contrast += self.glcm[i, j] * (i - j) ** 2
        return contrast

    def calculate_glcm_dissimilarity(self):
        dissimilarity = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                dissimilarity += self.glcm[i, j] * np.abs(i - j)
        return dissimilarity

    def calculate_glcm_entropy(self):
        eps = 1e-10
        entropy = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                entropy -= self.glcm[i, j] * np.log10(self.glcm[i, j] + eps)
        return entropy

    def calculate_glcm_energy(self):
        energy = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                energy += self.glcm[i, j] ** 2
        return energy

    def calculate_glcm_correlation(self):
        mean = self.calculate_glcm_mean()
        variance = self.calculate_glcm_variance()
        correlation = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                correlation += ((i - mean) * (j - mean) * (self.glcm[i, j] ** 2)) / variance
        return correlation

    def calculate_glcm_auto_correlation(self):
        auto_correlation = np.zeros((self.glcm.shape[2], self.glcm.shape[3]), dtype=np.float32)
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                auto_correlation += self.glcm[i, j] * i * j
        return auto_correlation