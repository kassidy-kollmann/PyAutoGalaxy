from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFWSph

from astropy import units

import numpy as np
import warnings
from autogalaxy import cosmology as cosmo

from scipy.integrate import quad


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


class gNFWVirialMassConcSph_BACKUP(gNFWSph):
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
    ):
        """
        Spherical gNFW profile initialized with the virial mass and concentration of the halo. 
        
        The virial radius of the halo is defined as: r_vir = (3*M_vir/4*pi*virial_overdens*critical_density)^1/3.
        
        If the virial_overdens parameter is set to 0, the virial overdensity of Bryan & Norman (1998) will be used.

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
            
        """
        self.virial_mass = virial_mass
        self.concentration = concentration
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.inner_slope = inner_slope

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
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )