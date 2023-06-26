#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import meep as mp
import math
import cmath
from scipy.special import jv

from datetime import datetime
from typing import Iterable, Type

from beams import Beam3D, CS1Bessel, CS2Bessel, LEBessel, LMBessel, TEBessel, TMBessel, ZRotatedBeam

print("Meep version:", mp.__version__)
print("\nStart time:", datetime.now())

# Media parameters
n1 = 1.0
n2 = 1.0
chi2 = 0.0
chi3 = 0.1

## angle of incidence (with respect to -z)
#chi_deg = 0.0

# Beam parameters
m_charge = 0        # topological charge
zeta_deg = 40       # axicon angle (deg)
r_w = 1             # distance to interface (wavelengths)

"""Define beams and amplitudes
First item in each tuple is a Beam3D class. An instance will be constructed with the args from all_beam_args
Second is amplitude of beam
Third is rotation of beam in degrees
"""
beams_and_params: Iterable[tuple[Type[Beam3D], complex, float]] = [
    (TEBessel,  0, 0),
    (TMBessel,  0, 0),
    (LEBessel,  1, 0),
    (LMBessel,  0, 0),
    (CS1Bessel, 0, 0),
    (CS2Bessel, 0, 0),
]

# MEEP parameters
sx = 8  # size of cell including PML in x-direction (um)
sy = 8  # size of cell including PML in y-direction (um)
sz = 4   # size of cell including PML in z-direction (um)
pml_thickness = 1.0   # thickness of PML layer (um)

lam = 1.064     # vacuum wavelength of source (um)

runtime = 20    # run time (in source periods)
pixels = 10     # pixels/um


freq = 1/lam    # vacuum frequency of source
k_vac = 2 * math.pi * freq  # vacuum wavenumber (um^-1)

resolution_factor = 1
if abs(chi3) > 0:
    resolution_factor = 3
elif abs(chi2) > 0:
    resolution_factor = 2
resolution = math.ceil(pixels * (n1 if n1 > n2 else n2) * resolution_factor * freq)
Courant = (n1 if n1 < n2 else n2) / 3

pml_layers = [mp.PML(pml_thickness)]

#chi_rad = math.radians(chi_deg)

zeta_rad = math.radians(zeta_deg)
kt = n1 * k_vac * math.cos(zeta_rad)
kl = n1 * k_vac * math.sin(zeta_rad)

# Flux regions
flux_refl_z = 1
flux_tran_z = 1
flux_region_size_x = sx / 4
flux_region_size_y = sy / 4

# Frequencies for flux regions
flux_freq_min = freq / 2.0
flux_freq_max = 4.0 * freq
flux_num_freqs = 600

cell = mp.Vector3(sx,sy,sz)
default_material = mp.Medium(index=n1)
refraction_material = mp.Medium(index=n2, chi2=chi2, chi3=chi3)
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, sz),
                     center=mp.Vector3(0, 0, sz/2),
                     e1=mp.Vector3(1, 0, 0),
                     e2=mp.Vector3(0, 1, 0),
                     e3=mp.Vector3(0, 0, 1),
                     material=refraction_material)]
                     
#geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, sz),
#                     center=mp.Vector3(0, 0, sz/(2 * math.cos(chi_rad))),
#                     e1=mp.Vector3(math.cos(chi_rad), 0, math.sin(chi_rad)),
#                     e2=mp.Vector3(0, 1, 0),
#                     e3=mp.Vector3(-math.sin(chi_rad), 0, math.cos(chi_rad)),
#                     material=refraction_material)]
          
force_complex_fields = False          # default: True
eps_averaging = True                  # default: True

print("\nExpected field output file size:",
          round(8*(sx*sy*sz*resolution**3)/(1024**2)), "MiB")

sources = []

def make_source(component, amplitude: complex, amp_func) -> mp.Source:
    """Makes source at common position offset from tilted medium"""
    return mp.Source(src=mp.ContinuousSource(frequency=freq, width=0.5),
                    component=component,
                    amplitude=amplitude,
                    size=mp.Vector3(sx, sy, 0),
                    center=mp.Vector3(0, 0, -r_w),
                    amp_func=amp_func
                    )

def make_beam_sources(beam: Beam3D, amplitude: complex) -> tuple[mp.Source, mp.Source, mp.Source]:
    """Make three sources for each component of the given beam
    Returns the three sources for the x, y, and z components in that order
    """
    return (
        make_source(mp.Ex, amplitude, beam.x),
        make_source(mp.Ey, amplitude, beam.y),
        make_source(mp.Ez, amplitude, beam.z),
    )

# Arguments to all beams
all_beam_args = {
    "n1": n1,
    "k_vac": k_vac,
    "zeta_rad": zeta_rad,
    "m_charge": m_charge,
}

for beam_type, amplitude, phi in beams_and_params:
    if amplitude == 0:
        continue
    sources.extend(make_beam_sources(ZRotatedBeam(beam_type(**all_beam_args), math.radians(phi)),
                                     amplitude))


progress_interval = 20
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    symmetries=[],
                    default_material=default_material,
                    Courant=Courant,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    force_complex_fields=force_complex_fields,
                    eps_averaging=eps_averaging,
                    progress_interval=progress_interval
                    )

sim.use_output_directory()   # put output files in a separate folder

# By default, flux region direction is in positive coordinate direction
# So these are in positive z direction
flux_refl = sim.add_flux(
    0.5 * (flux_freq_min + flux_freq_max),
    flux_freq_max - flux_freq_min,
    flux_num_freqs,
    mp.FluxRegion(
        direction=mp.Z,
        center=mp.Vector3(z=-flux_refl_z),
        size=mp.Vector3(0*flux_region_size_x, 0*flux_region_size_y, 0)
    )
)

flux_tran = sim.add_flux(
    0.5 * (flux_freq_min + flux_freq_max),
    flux_freq_max - flux_freq_min,
    flux_num_freqs,
    mp.FluxRegion(
        direction=mp.Z,
        center=mp.Vector3(z=flux_tran_z),
        size=mp.Vector3(0*flux_region_size_x, 0*flux_region_size_y, 0)
    )
)

run_args = [# mp.at_beginning(mp.output_epsilon),    # output of dielectric function
            mp.at_end(mp.output_efield_x),         # output of E_x component
            # mp.at_end(mp.output_efield_y),         # output of E_y component
            # mp.at_end(mp.output_efield_z),         # output of E_z component
            # mp.at_every(0.2, mp.in_volume(mp.Volume(size=mp.Vector3(sx, 0, sz), center=mp.Vector3(0, 0, 0)),
            #                                mp.output_png(mp.Ex, "-S6 -T -Zc dkbluered"))),
            ]

sim.run(*run_args, until=runtime)

#sim.display_fluxes(flux_refl, flux_tran)
#sim.save_flux("flux_refl", flux_refl)
#sim.save_flux("flux_tran", flux_tran)

print("\nEnd time:", datetime.now())

import numpy as np

flux_tran_out = np.asarray(mp.get_fluxes(flux_tran))
flux_freq_out = np.asarray(mp.get_flux_freqs(flux_tran))
np.savetxt('flux_tran_out.txt', np.c_[flux_freq_out, flux_tran_out], delimiter=',')

flux_refl_out = np.asarray(mp.get_fluxes(flux_refl))
flux_freq_out = np.asarray(mp.get_flux_freqs(flux_refl))
np.savetxt('flux_refl_out.txt', np.c_[flux_freq_out, flux_refl_out], delimiter=',')
