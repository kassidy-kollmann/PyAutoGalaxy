from typing import List, Union

from autoarray.plot.mat_wrap import get_visuals

from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D

from autogalaxy.util import error_util

from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile


class GetVisuals1D(get_visuals.GetVisuals1D):
    def __init__(self, include: Include1D, visuals: Visuals1D):

        super().__init__(include=include, visuals=visuals)

    def via_light_obj_from(self, light_obj) -> Visuals1D:
        """
        Extracts from the `LightProfile` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LightProfilePlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """

        if light_obj is None:
            return self.visuals

        half_light_radius = self.get(
            "half_light_radius", value=light_obj.half_light_radius
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius
        )

    def via_light_obj_list_from(
        self, light_obj_list: List[LightProfile], low_limit: float
    ) -> Visuals1D:
        """
        Extracts from the `LightProfile` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LightProfilePlotter` the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the `LightProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """

        if self.include.half_light_radius:

            half_light_radius_list = [
                light_profile.half_light_radius for light_profile in light_obj_list
            ]

            if None in half_light_radius_list:

                half_light_radius = None
                half_light_radius_errors = None

            else:

                half_light_radius, half_light_radius_errors = error_util.value_median_and_error_region_via_quantile(
                    value_list=half_light_radius_list, low_limit=low_limit
                )

        else:

            half_light_radius = None
            half_light_radius_errors = None

        half_light_radius = self.get("half_light_radius", value=half_light_radius)
        half_light_radius_errors = self.get(
            "half_light_radius", value=half_light_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius,
            half_light_radius_errors=half_light_radius_errors,
        )

    def via_mass_obj_from(self, mass_obj, grid) -> Visuals1D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LensingObjProfilePlotter` the following 1D attributes can be extracted for plotting:

        - einstein_radius: the Einstein radius of the `MassProfile`.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """

        if mass_obj is None:
            return self.visuals

        if self.include.einstein_radius:
            einstein_radius = mass_obj.einstein_radius_from(grid=grid)
        else:
            einstein_radius = None

        einstein_radius = self.get("einstein_radius", value=einstein_radius)

        return self.visuals + self.visuals.__class__(einstein_radius=einstein_radius)

    def via_mass_obj_list_from(
        self, mass_obj_list, grid, low_limit: float
    ) -> Visuals1D:
        """
        Extracts from the `MassProfile` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `MassProfilePlotter` the following 1D attributes can be extracted for plotting:

        - einstein_radius: the radius containing 50% of the `MassProfile`'s total integrated luminosity.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """

        if self.include.einstein_radius:

            einstein_radius_list = [
                mass_profile.einstein_radius_from(grid=grid)
                for mass_profile in mass_obj_list
            ]

            einstein_radius, einstein_radius_errors = error_util.value_median_and_error_region_via_quantile(
                value_list=einstein_radius_list, low_limit=low_limit
            )

        else:

            einstein_radius = None
            einstein_radius_errors = None

        einstein_radius = self.get("einstein_radius", value=einstein_radius)
        einstein_radius_errors = self.get(
            "einstein_radius", value=einstein_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            einstein_radius=einstein_radius,
            einstein_radius_errors=einstein_radius_errors,
        )


class GetVisuals2D(get_visuals.GetVisuals2D):
    def __init__(self, include: Include2D, visuals: Visuals2D):

        super().__init__(include=include, visuals=visuals)

    def via_light_obj_from(self, light_obj: LightProfile, grid) -> Visuals2D:
        """
        Extracts from the `LightProfile` attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LightProfilePlotter` the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_via_mask = self.via_mask_from(mask=grid.mask)

        if isinstance(light_obj, LightProfile):

            light_profile_centres = self.get(
                "light_profile_centres", Grid2DIrregular(grid=[light_obj.centre])
            )

        else:

            light_profile_centres = self.get(
                "light_profile_centres",
                light_obj.extract_attribute(cls=LightProfile, attr_name="centre"),
            )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(light_profile_centres=light_profile_centres)
        )

    def via_mass_obj_from(self, mass_obj, grid) -> Visuals2D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_via_mask = self.via_mask_from(mask=grid.mask)

        if isinstance(mass_obj, MassProfile):

            mass_profile_centres = self.get(
                "mass_profile_centres", Grid2DIrregular(grid=[mass_obj.centre])
            )

        else:

            mass_profile_centres = self.get(
                "mass_profile_centres",
                mass_obj.extract_attribute(cls=MassProfile, attr_name="centre"),
            )

        critical_curves = self.get(
            "critical_curves",
            mass_obj.critical_curves_from(grid=grid),
            "critical_curves",
        )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(
                mass_profile_centres=mass_profile_centres,
                critical_curves=critical_curves,
            )
        )

    def via_light_mass_obj_from(self, light_mass_obj, grid) -> Visuals2D:
        """
        From an object that contains both light and lensing attributes (e.g. a `Galaxy`, `Plane`), gets the
        attributes that can be plotted and returns them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted.

        From a light and lensing object the following attributes can be extracted for plotting:

        - mask: the mask of the grid used to plot the 2D quantities of the object.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critcal curves: the critical curves of all mass profile combined.

        Parameters
        ----------
        light_mass_obj
            The object which has `LightProfile` objects and / or `MassProfile` objects.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_2d = self.via_mass_obj_from(mass_obj=light_mass_obj, grid=grid)
        visuals_2d.mask = None

        visuals_with_grid = self.visuals.__class__(grid=self.get("grid", grid))

        return (
            visuals_2d
            + visuals_with_grid
            + self.via_light_obj_from(light_obj=light_mass_obj, grid=grid)
        )

    def via_fit_from(self, fit: FitImaging) -> Visuals2D:

        visuals_2d_via_fit = super().via_fit_from(fit=fit)

        visuals_2d_via_light_mass_obj = self.via_light_mass_obj_from(
            light_mass_obj=fit.plane, grid=fit.grid
        )

        return visuals_2d_via_fit + visuals_2d_via_light_mass_obj
