from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFW

from astropy import units

import numpy as np
import warnings
from autogalaxy import cosmology as cosmo

from scipy import special
from scipy.integrate import quad

import autoarray as aa


def kappa_s_and_scale_radius(
    cosmology, virial_mass, concentration, virial_overdens, redshift_object, redshift_source, inner_slope
):

    critical_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc**3)
    ).value

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)
    
    if virial_overdens == 0:
        x = cosmology.Om(redshift_object) - 1
        virial_overdens = 18*np.pi**2 + 82*x - 39*x**2 # Bryan & Norman (1998)

    virial_radius = (
        virial_mass / (virial_overdens * critical_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r_vir
    
    scale_radius_kpc = virial_radius / concentration  # scale radius in kpc
    
    ##############################
    def integrand(r):
        return (r**2 / r**inner_slope) * (1 + r/scale_radius_kpc)**(inner_slope-3) 
        
    de_c = ((virial_overdens / 3.0) * (virial_radius**3 / scale_radius_kpc**inner_slope) 
            / quad(integrand,0,virial_radius)[0]) # rho_c
    ##############################
    
    rho_s = critical_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, virial_radius, virial_overdens


class gNFWVirialMassConcSph_modify_deflections(gNFW):
    def __init__(
        self,
        cosmology: cosmo.LensingCosmology = cosmo.Planck15(),
        centre: Tuple[float, float] = (0.0, 0.0),
        virial_mass: float = 1e12,
        concentration: float = 10,
        virial_overdens: float = 0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        inner_slope: float = 1.0,
        cut_off_R: float = 1000.0,
    ):
        """
        Spherical gNFW profile initialized with the virial mass and concentration of the halo. 
        
        The virial radius of the halo is defined as: r_vir = (3*M_vir/4*pi*virial_overdens*critical_density)^1/3.
        
        If the virial_overdens parameter is set to 0, the virial overdensity of Bryan & Norman (1998) will be used.
        
        Unlike the other gNFWVirialMassConcSph function, this one's parent is not gNFWSph, it is gNFW, since I have added all of the functions from gNFWSph below.

        Parameters
        ----------
        cosmology
            The cosmology to use in calculations.
        centre
            The (y,x) arc-second coordinates of the profile centre.
        virial_mass
            The virial mass of the dark matter halo.
        concentration
            The concentration of the dark matter halo.
        virial_overdens
            The virial overdensity.
        redshift_object
            Lens redshift.
        redshift_source
            Source redshift.
        inner_slope
            The inner slope of the dark matter halo profile.
        cut_off_R
            The 2D distance from the subhalo at which to remove deflections.
            
        """
        self.virial_mass = virial_mass
        self.concentration = concentration
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.inner_slope = inner_slope
        self.cut_off_R = cut_off_R

        (
            kappa_s,
            scale_radius,
            virial_radius,
            virial_overdens,
        ) = kappa_s_and_scale_radius(
            cosmology=cosmology,
            virial_mass=virial_mass,
            concentration=concentration,
            virial_overdens=virial_overdens,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            inner_slope=inner_slope
        )
        
        self.virial_radius = virial_radius
        self.virial_overdens = virial_overdens
        
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
        
        
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        print('In new function')
        
        deflections = self.deflections_2d_via_mge_from(grid=grid)
        
        R_grid = self.radial_grid_from(grid)
        
        for i in range(grid.shape[0]):
            
            if R_grid[i] >= self.cut_off_R:
                deflections[i] = [0,0]

        return deflections

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.radial_grid_from(grid))
        
        R_grid = self.radial_grid_from(grid)

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            
            if R_grid[i] <= self.cut_off_R:
        
                deflection_grid[i] = np.multiply(
                    4.0 * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta[i])
                )
                
            # otherwise, leave the deflection as zero

        return self._cartesian_grid_via_radial_from(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y**2)) / y)

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(
            self.deflection_integrand,
            a=0.0,
            b=1.0,
            args=(eta, self.inner_slope),
            epsrel=1.49e-6,
        )[0]
        return eta ** (2 - self.inner_slope) * (
            (1.0 / (3 - self.inner_slope))
            * special.hyp2f1(
                3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope, -eta
            )
            + integral_y_2
        )
