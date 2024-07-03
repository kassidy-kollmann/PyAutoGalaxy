import numpy as np

import autoarray as aa

from autogalaxy.ellipse.dataset_ellipse import DatasetEllipse
from autogalaxy.ellipse.ellipse import Ellipse


class FitEllipse(aa.FitDataset):
    def __init__(self, dataset: DatasetEllipse, ellipse: Ellipse):
        """
        A fit to a `DatasetEllipse` dataset, using a model image to represent the observed data and noise-map.

        Parameters
        ----------
        dataset
            The dataset containing the signal and noise-map that is fitted.

        """
        super().__init__(dataset=dataset)

        self.ellipse = ellipse

    @property
    def angles_array(self):
        n = np.min([500, int(self.circular_radius)])

        return np.linspace(0.0, 2.0 * np.pi, n)

    @property
    def points(self):
        x = self.ellipse.x_from(angles_array=self.angles_array)
        y = self.ellipse.y_from(angles_array=self.angles_array)

        idx = np.logical_or(np.isnan(x), np.isnan(y))
        if np.sum(idx) > 0.0:
            raise NotImplementedError()

        return np.stack(arrays=(y, x), axis=-1)

    @property
    def data(self) -> aa.Array1D:
        return aa.Array1D.no_mask(values=self.dataset.data_interp(self.points))

    @property
    def noise_map(self):
        return self.dataset.noise_map(self.points)

    @property
    def signal_to_noise_map(self):
        return self.data / self.noise_map

    @property
    def model_data(self):
        return self.data - np.nanmean(self.data)

    @property
    def residual_map(self):
        return self.model_data

    @property
    def normalized_residual_map(self):
        normalized_residual_map = (self.model_data) / self.noise_map

        # NOTE:
        idx = np.logical_or(
            np.isnan(normalized_residual_map), np.isinf(normalized_residual_map)
        )
        normalized_residual_map[idx] = 0.0

        return normalized_residual_map

    @property
    def chi_squared_map(self):
        return (self.normalized_residual_map) ** 2.0

    @property
    def chi_squared(self):
        return np.sum(self.chi_squared_map)

    @property
    def noise_normalization(self):
        return np.sum(np.log(2 * np.pi * self.noise_map**2.0))

    @property
    def log_likelihood(self):
        return -0.5 * (self.chi_squared + self.noise_normalization)

    @property
    def figure_of_merit(self) -> float:
        return self.log_likelihood
