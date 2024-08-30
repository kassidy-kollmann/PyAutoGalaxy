from typing import Tuple

import autoarray as aa
from astropy import units as u
from astropy import constants

import numpy as np
import warnings
from scipy.integrate import quad
from mpmath import nsum, inf, factorial, gamma
import time

from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.dark.abstract import DarkProfile

def calculate_quantities_for_Dekel(
    virial_mass_Msol,DekelConcentration,DekelSlope,sph_overdens,redshift_object,redshift_source
):
    c = DekelConcentration
    a = DekelSlope
    
    cosmology = Planck15()

    critical_density_Msol_kpc3 = (
        cosmology.critical_density(redshift_object).to(u.solMass / u.kpc**3)
    ).value
    
    critical_surface_density_Msol_kpc2 = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )
    
    virial_radius_kpc = (
        virial_mass_Msol / (sph_overdens * critical_density_Msol_kpc3 * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )
    
    r_c_kpc = virial_radius_kpc / c
    
    rho_c_Msol_kpc3 = (3-a) * pow(c,a) * pow(1+pow(c,0.5),6-2*a) * virial_mass_Msol / (4*np.pi*pow(virial_radius_kpc,3))
    
    Beta_for_kappa_0 = gamma(2-2*a)*gamma(5)/(gamma(2-2*a+5))
    
    kappa_0 = 4 * rho_c_Msol_kpc3 * r_c_kpc * Beta_for_kappa_0 / critical_surface_density_Msol_kpc2 # dimensionless
    
    return rho_c_Msol_kpc3, r_c_kpc, virial_radius_kpc, kappa_0, critical_surface_density_Msol_kpc2
    

class DekelSph(MassProfile, DarkProfile):
    
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        virial_mass_Msol: float = 1e12,
        DekelConcentration: float = 5,
        DekelSlope: float = 5,
        sph_overdens: float = 200,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        method: str = 'letter'
    ):
        """
        The spherical Dekel profile, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        virial_mass_Msol
            The virial mass of the dark matter halo.
        c_2
            The conventional concentration parameter that relates the radius at which the logarithmic density slope 
            equals -2, r_2, to the virial radius. 
            r_2 = r_vir/c_2
            For a NFW profile r_2 is the same as the scale radius.
            No longer inputting this
        DekelConcentration
            The Dekel concentration, c, relates the intermediate radius, r_c, to the virial radius.
            r_c = r_vir/c.
            The conventional concentration parameter, c_2, is related to the Dekel concentration as follows:
            c_2 = c*(1.5 / (2-a))**2, where a is the Dekel slope.            
        DekelSlope
            The negative of the logarithmic density slope in the center of the halo.
            a = -dln(rho)/dln(r) for r --> 0.
            A positive density imposes a <= 3. The Dekel slope can be negative for realistic profiles.
        sph_overdens
            The spherical overdensity.
        redshift_object
        redshift_source
        method
            'a' for analytic expressions, 'i' for numerical integration
        """
        
        self.virial_mass_Msol = virial_mass_Msol
        self.DekelConcenteration = DekelConcentration
        self.DekelSlope = DekelSlope
        self.sph_overdens = sph_overdens
        self.method = method
        
        self.c_2 = DekelConcentration*(1.5/(2-DekelSlope))**2
        
        (
            rho_c_Msol_kpc3,
            r_c_kpc,
            virial_radius_kpc,
            kappa_0,
            lens_crit_dens_Msol_kpc2
        ) = calculate_quantities_for_Dekel(
            virial_mass_Msol=virial_mass_Msol,
            DekelConcentration=DekelConcentration,
            DekelSlope=DekelSlope,
            sph_overdens=sph_overdens,         
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
        
        self.rho_c_Msol_kpc3 = rho_c_Msol_kpc3
        self.r_c_kpc = r_c_kpc
        self.virial_radius_kpc = virial_radius_kpc
        self.kappa_0 = kappa_0
        self.lens_crit_dens_Msol_kpc2 = lens_crit_dens_Msol_kpc2
        
        cosmology = Planck15()
        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)
        self.kpc_per_arcsec = kpc_per_arcsec
        
        self.r_c_arcsec = r_c_kpc / kpc_per_arcsec # arcsec
        
        super().__init__(centre=centre)
    
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        
        if self.method == 'a':
            return self.deflections_2d_via_analytic_from(grid=grid)

        elif self.method == 'i':
            return self.deflections_2d_via_integral_from(grid=grid)
    
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_analytic_from(self, grid: aa.type.Grid2DLike):
    
        X = (1.0 / self.r_c_arcsec) * self.radial_grid_from(grid) # dimensionless
        
        gamma_part_sum1 = []
        exp_part_sum1 = []
        gamma_part_sum2 = []
        exp_part_sum2 = []

        # iterate over k values
        for k in range(201):
            gamma_part_sum1.append((-1)**k/factorial(k) * gamma(-2*self.DekelSlope + 2 - 4*k) * gamma(5 + 4*k) / (gamma(0.5 - k)*(k+1)))
            exp_part_sum1.append(2*k + 1)
        
            gamma_part_sum2.append((-1)**k/(4*factorial(k)) * gamma(-1/2 + self.DekelSlope/2 - k/4) * gamma(7 - 2*self.DekelSlope + k) / (gamma(self.DekelSlope/2 - k/4)*(1.5 - self.DekelSlope/2 + k/4)))
            exp_part_sum2.append((2 - self.DekelSlope + k/2))
    
        gamma_part_sum1 = np.array(gamma_part_sum1)
        exp_part_sum1 = np.array(exp_part_sum1)
        gamma_part_sum2 = np.array(gamma_part_sum2)
        exp_part_sum2 = np.array(exp_part_sum2)
        
        def sum1_func_no_gammas(k,X_val):
            return gamma_part_sum1[int(k)] * X_val**exp_part_sum1[int(k)]
        
        def sum2_func_no_gammas(k,X_val):
            return gamma_part_sum2[int(k)] * X_val**exp_part_sum2[int(k)]
        
        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            sum1_at_each_k = []
            sum2_at_each_k = []
            for k in range(201):
                sum1_at_each_k.append(sum1_func_no_gammas(k,X_val=X[i]))
                sum2_at_each_k.append(sum2_func_no_gammas(k,X_val=X[i]))
            sum1_for_all_k = sum(sum1_at_each_k)
            sum2_for_all_k = sum(sum2_at_each_k)
            
            deflection_grid[i] = 4*np.sqrt(np.pi)*self.rho_c_Msol_kpc3*self.r_c_kpc/(self.lens_crit_dens_Msol_kpc2*gamma(7-2*self.DekelSlope)) * (sum1_for_all_k + sum2_for_all_k)

        return self._cartesian_grid_via_radial_from(grid, deflection_grid)
    
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(self, grid: aa.type.Grid2DLike):
        
        R = self.radial_grid_from(grid) # arcsec
        
        deflection_grid = np.zeros(grid.shape[0])
        
        for i in range(grid.shape[0]):
            deflection_grid[i] = (1 / (R[i]*self.kpc_per_arcsec * np.pi*self.lens_crit_dens_Msol_kpc2*self.r_c_kpc))* 2*np.pi* quad(
                self.cumulative_mass_integrand,
                a=0,
                b=R[i]*self.kpc_per_arcsec, 
            )[0]
        
        return self._cartesian_grid_via_radial_from(grid, deflection_grid)
        
    def cumulative_mass_integrand(self,R_prime_kpc):

        cumulative_mass = R_prime_kpc * 2 * self.rho_c_Msol_kpc3 * self.r_c_kpc * quad(
            self.surface_density_integrand,
            a=R_prime_kpc/self.r_c_kpc,
            b=np.inf,
            args=(
                R_prime_kpc/self.r_c_kpc,
                self.DekelSlope,
            )
        )[0]
        
        return cumulative_mass
        
 
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        if self.method == 'a':
            return self.convergence_2d_via_analytic_from(grid=grid)

        elif self.method == 'i':
            return self.convergence_2d_via_integral_from(grid=grid)
        
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_via_analytic_from(self, grid: aa.type.Grid2DLike):
        
        X = (1.0 / self.r_c_arcsec) * self.radial_grid_from(grid) # dimensionless
        
        gamma_part_sum1 = []
        exp_part_sum1 = []
        gamma_part_sum2 = []
        exp_part_sum2 = []

        # iterate over k values
        for k in range(201):
            gamma_part_sum1.append((-1)**k/factorial(k) * gamma(-2*self.DekelSlope + 2 - 4*k) * gamma(5 + 4*k) / gamma(0.5 - k))
            exp_part_sum1.append(2*k)
        
            gamma_part_sum2.append((-1)**k/(4*factorial(k)) * gamma(-1/2 + self.DekelSlope/2 - k/4) * gamma(7 - 2*self.DekelSlope + k) / gamma(self.DekelSlope/2 - k/4))
            exp_part_sum2.append((1 - self.DekelSlope + k/2))
    
        gamma_part_sum1 = np.array(gamma_part_sum1)
        exp_part_sum1 = np.array(exp_part_sum1)
        gamma_part_sum2 = np.array(gamma_part_sum2)
        exp_part_sum2 = np.array(exp_part_sum2)
        
        def sum1_func_no_gammas(k,X_val):
            return gamma_part_sum1[int(k)] * X_val**exp_part_sum1[int(k)]
        
        def sum2_func_no_gammas(k,X_val):
            return gamma_part_sum2[int(k)] * X_val**exp_part_sum2[int(k)]
        
        
        convergence_grid = np.zeros(grid.shape[0])
        
        for i in range(grid.shape[0]):
            sum1_at_each_k = []
            sum2_at_each_k = []
            for k in range(201):
                sum1_at_each_k.append(sum1_func_no_gammas(k,X_val=X[i]))
                sum2_at_each_k.append(sum2_func_no_gammas(k,X_val=X[i]))
            sum1_for_all_k = sum(sum1_at_each_k)
            sum2_for_all_k = sum(sum2_at_each_k)
            
            convergence_grid_val = 4*np.sqrt(np.pi)*self.rho_c_Msol_kpc3*self.r_c_kpc/(self.lens_crit_dens_Msol_kpc2*gamma(7-2*self.DekelSlope)) * (sum1_for_all_k + sum2_for_all_k)
            if convergence_grid_val > 10:
                convergence_grid[i] = 0
            else: 
                convergence_grid[i] = convergence_grid_val
            
        return convergence_grid
        
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_via_integral_from(self, grid: aa.type.Grid2DLike):
    
        X = (1.0 / self.r_c_arcsec) * self.radial_grid_from(grid) # dimensionless
        
        convergence_grid = np.zeros(grid.shape[0])
        
        for i in range(grid.shape[0]):
            convergence_grid[i] = 2 * self.rho_c_Msol_kpc3 * self.r_c_kpc / self.lens_crit_dens_Msol_kpc2 * quad(
                self.surface_density_integrand,
                a=X[i],
                b=np.inf,
                args=(
                    X[i],
                    self.DekelSlope,
                ),   
            )[0]  # dimensionless
        
        return convergence_grid
    
    @staticmethod
    def surface_density_integrand(x,X,DekelSlope):
        return pow(x,1.-DekelSlope)*pow(1.+pow(x,0.5),2.*DekelSlope-7)/np.sqrt(pow(x,2)-pow(X,2)) # dimensionless
        
        
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        
        X = (1.0 / self.r_c_arcsec) * self.radial_grid_from(grid) # dimensionless
        
        def sum1_func(k,X_val):
            return (-1)**k/factorial(k) * gamma(-2*self.DekelSlope + 2 - 4*k)*gamma(5 + 4*k)/(gamma(0.5-k) * (k+1)**2) * X_val**(2*k + 2)
            
        def sum2_func(k,X_val):
            return (-1)**k/(4*factorial(k)) * gamma(-1/2 + self.DekelSlope/2 - k/4)*gamma(7 - 2*self.DekelSlope + k)/(gamma(self.DekelSlope/2 - k/4) * (3/2 - self.DekelSlope/2 + k/4)**2) * X_val**(3 - self.DekelSlope + k/2)
        
        potential_grid = np.zeros(grid.shape[0])
        
        for i in range(grid.shape[0]):
            potential_grid[i] = 2*np.sqrt(np.pi)*self.rho_c_Msol_kpc3*self.r_c_kpc/(self.lens_crit_dens_Msol_kpc2*gamma(7-2*self.DekelSlope)) * float(nsum(lambda k: sum1_func(k,X_val=X[i]),[0,inf])
                + nsum(lambda k: sum2_func(k,X_val=X[i]),[0,inf]))
            
        return self._cartesian_grid_via_radial_from(grid, potential_grid)
    