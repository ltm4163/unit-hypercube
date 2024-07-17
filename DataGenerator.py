import numpy as np
import math


class DataGenerator:
    def __init__(self, agents):
        self.agents = agents

    def truncated_normal(self, n, minval=0, maxval=1):
        return np.clip(np.random.normal(0.85, 0.05, (n, self.agents)), minval, maxval)

    def linear(self, n):
        data = []
        for _ in range(self.agents):
            data.append(np.linspace(0.20, 0.92, n) + np.random.normal(-0.05, 0.05, n))
        return np.array(data).T

    def sinusoidal(self, n):
        x = np.linspace(0, 1, n)
        datasets = [x]
        for _ in range(self.agents - 1):
            noise = np.random.normal(scale=0.07, size=n)
            dataset = [0.5 + 0.2 * math.sin(2 * math.pi * i / n) for i in range(n)]
            dataset += noise  # Add noise to dataset
            datasets.append(dataset)

        # Combine datasets into a num_points x dimension array
        combined_dataset = np.vstack(datasets).T
        return combined_dataset

    def gen_outliers(self, n, finlier):
        cluster_points = finlier(n)
        outliers = []
        while len(outliers) < n:
            point = np.random.rand(self.agents)
            distances = np.linalg.norm(cluster_points - point, axis=1)
            if np.all(distances > 0.05):  # Adjust threshold as needed
                outliers.append(point)
        return np.array(outliers)

    def generate_test_data(self, n, finlier, contamination):
        n_outliers = int(n * contamination)
        n_inliers = n - n_outliers
        outliers = self.gen_outliers(n_outliers, finlier)
        inliers = finlier(n_inliers)
        return np.vstack((outliers, inliers))

    def generate_patient_vital_signs(self, n):
        train_data = self.truncated_normal(n)
        test_data = self.generate_test_data(n, self.truncated_normal, 0.05)
        return train_data, test_data

    def generate_traffic_flow(self, n):
        train_data = self.linear(n)
        test_data = self.generate_test_data(n, self.linear, 0.05)
        return train_data, test_data

    def generate_political_influence(self, n):
        train_data = self.sinusoidal(n)
        test_data = self.generate_test_data(n, self.sinusoidal, 0.05)
        return train_data, test_data

    def gen_static_data(self, n):
        health_data, health_test = self.generate_patient_vital_signs(n)
        np.savetxt("test/test1.csv", np.array(health_test), delimiter=',')
        np.savetxt("train/train1.csv", health_data, delimiter=',')

        traffic_data, traffic_test = self.generate_traffic_flow(n)
        np.savetxt("test/test2.csv", traffic_test, delimiter=',')
        np.savetxt("train/train2.csv", traffic_data, delimiter=',')

        political_data, political_test = self.generate_political_influence(n)
        np.savetxt("test/test3.csv", political_test, delimiter=',')
        np.savetxt("train/train3.csv", political_data, delimiter=',')
