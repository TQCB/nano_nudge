from typing import Optional

import numpy as np
import plotly.express as px
from scipy.stats import norm

class KDE:
    def __init__(
            self,
            data: np.ndarray,
            bandwidth: Optional[float]=None,
            min_bandwidth: float=1,
    ):
        self.data = data
        self.n = len(self.data)

        if bandwidth is None:
            std_dev = np.std(self.data)

            if std_dev == 0:
                std_dev = min_bandwidth

            calculated_bandwidth = self.n**(-1/5) * std_dev
            self.bandwidth = max(calculated_bandwidth, min_bandwidth)
        else:
            self.bandwidth = bandwidth

        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")

    def sample(self, num_samples):
        # We randomly select from our initial data points
        chosen_indices = np.random.choice(self.n, size=num_samples, replace=True)
        chosen_data_points = self.data[chosen_indices]

        # And we draw a normally distributed sample from the data point
        new_samples = norm.rvs(loc=chosen_data_points, scale=self.bandwidth)
        return new_samples

    def evaluate(self, query_points):
        """
        Evaluates the KDE at given query_points.
        Points are evaluated directly, there is no precomputation in this implementation.
        """
        query_points = np.asarray(query_points)

        density_values = np.zeros_like(query_points, dtype=float)
        for x_i in self.data:
            # use pdf to create a normal distribution of mu x_i and sigma bandwidth
            density_values += norm.pdf(query_points, loc=x_i, scale=self.bandwidth)
        return density_values / self.n
    
    def plot(self, title=None):
        x = np.arange(-10, 10, 0.1)
        y = self.evaluate(x)

        fig = px.scatter(x=x, y=y, title=title)
        fig.show()
    
    def __call__(self, query_points):
        return self.evaluate(query_points)