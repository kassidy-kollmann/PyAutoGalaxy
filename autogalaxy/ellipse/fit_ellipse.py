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
    def data(self) -> aa.ArrayIrregular:
        """
        Returns the data values of the dataset that the ellipse fits, which are computed by overlaying the ellipse over
        the 2D data and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        Returns
        -------
        The data values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the data.
        """
        return aa.ArrayIrregular(values=self.dataset.data_interp(self.ellipse.points_from_major_axis))

    @property
    def noise_map(self) -> aa.ArrayIrregular:
        """
        Returns the noise-map values of the dataset that the ellipse fits, which are computed by overlaying the ellipse
        over the 2D noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        Returns
        -------
        The noise-map values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the noise-map.
        """
        return aa.ArrayIrregular(values=self.dataset.noise_map_interp(self.ellipse.points_from_major_axis))

    @property
    def signal_to_noise_map(self) -> aa.ArrayIrregular:
        """
        Returns the signal-to-noise-map of the dataset that the ellipse fits, which is computed by overlaying the ellipse
        over the 2D data and noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        Returns
        -------
        The signal-to-noise-map values of the ellipse fits, computed via a 2D interpolation of where
        the ellipse overlaps the data and noise-map.
        """
        return aa.ArrayIrregular(values=self.data / self.noise_map)

    @property
    def model_data(self) -> aa.ArrayIrregular:
        """
        Returns the model-data, which is the data values where the ellipse overlaps the data minus the mean
        value of these data values.

        By subtracting the mean of the data values from each data value, the model data quantifies how often there
        are large variations in the data values over the ellipse.

        For example, if every data value subtended by the ellipse are close to one another, the difference between
        the data values and the mean will be small.

        Conversely, if some data values are much higher or lower than the mean, the model data will be large.

        Returns
        -------
        The model data values of the ellipse fit, which are the data values minus the mean of the data values.
        """
        return aa.ArrayIrregular(values=self.data - np.nanmean(self.data))

    @property
    def residual_map(self):
        """
        Returns the residual-map of the fit, which is the data minus the model data and therefore the same
        as the model data.

        Returns
        -------
        The residual-map of the fit, which is the data minus the model data and therefore the same as the model data.
        """
        return self.model_data

    @property
    def normalized_residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the normalized residual-map of the fit, which is the residual-map divided by the noise-map.

        The residual map and noise map are computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `residual_map` and `noise_map` properties for more details.

        Returns
        -------
        The normalized residual-map of the fit, which is the residual-map divided by the noise-map.
        """

        normalized_residual_map = (self.model_data) / self.noise_map

        # NOTE:
        idx = np.logical_or(
            np.isnan(normalized_residual_map), np.isinf(normalized_residual_map)
        )
        normalized_residual_map[idx] = 0.0

        return aa.ArrayIrregular(values=normalized_residual_map)

    @property
    def chi_squared_map(self) -> aa.ArrayIrregular:
        """
        Returns the chi-squared-map of the fit, which is the normalized residual-map squared.

        The normalized residual map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `normalized_residual_map` property for more details.

        Returns
        -------
        The chi-squared-map of the fit, which is the normalized residual-map squared.
        """
        return aa.ArrayIrregular(values=self.normalized_residual_map ** 2.0)

    @property
    def chi_squared(self) -> float:
        """
        The sum of the chi-squared-map, which quantifies how well the model data represents the data and noise-map.

        The chi-squared-map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `chi_squared_map` property for more details.

        Returns
        -------
        The chi-squared of the fit.
        """
        return float(np.sum(self.chi_squared_map))

    @property
    def noise_normalization(self):
        """
        The noise normalization term of the log likelihood, which is the sum of the log noise-map values squared.

        Returns
        -------
        The noise normalization term of the log likelihood.
        """
        return np.sum(np.log(2 * np.pi * self.noise_map**2.0))

    @property
    def log_likelihood(self):
        """
        The log likelihood of the fit, which quantifies how well the model data represents the data and noise-map.

        Returns
        -------
        The log likelihood of the fit.
        """
        return -0.5 * (self.chi_squared)

    @property
    def figure_of_merit(self) -> float:
        """
        The figure of merit of the fit, which is passed by the `Analysis` class to the non-linear search to
        determine the best-fit solution.

        Returns
        -------
        The figure of merit of the fit.
        """
        return self.log_likelihood
