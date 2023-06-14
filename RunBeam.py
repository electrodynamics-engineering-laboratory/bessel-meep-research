#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import meep as mp
import math
import cmath
from scipy.special import jv

from datetime import datetime
from typing import Iterable, Type

from beams import Beam3D, CS1Bessel, CS2Bessel, LEBessel, LMBessel, TEBessel, TMBessel

print("Meep version:", mp.__version__)
print("\nStart time:", datetime.now())

# Media parameters
n1 = 1.0
n2 = 1.54

# angle of incidence (with respect to -z)
chi_deg = 15.0

# Beam parameters
m_charge = 2        # topological charge
zeta_deg = 70       # axicon angle (deg)
r_w = 3             # distance to interface (wavelengths)

"""Define beams and amplitudes
First item in each tuple is a Beam3D class. An instance will be constructed with the args from all_beam_args
Second is amplitude of beam
"""
beams_and_amps: Iterable[tuple[Type[Beam3D], complex]] = [
    (TEBessel,  0.5),
    (TMBessel,  0.5*1j),
    (LEBessel,  0),
    (LMBessel,  0),
    (CS1Bessel, 0),
    (CS2Bessel, 0),
]

# MEEP parameters
sx = 24  # size of cell including PML in x-direction (um)
sy = 24  # size of cell including PML in y-direction (um)
sz = 15   # size of cell including PML in z-direction (um)
pml_thickness = 1.0   # thickness of PML layer (um)

lam = 1.55      # vacuum wavelength of source (um)

runtime = 20    # run time (in source periods)
pixels = 15     # pixels/um


freq = 1/lam    # vacuum frequency of source
k_vac = 2 * math.pi * freq  # vacuum wavenumber (um^-1)

resolution = math.ceil(pixels * (n1 if n1 > n2 else n2) * freq)
Courant = (n1 if n1 < n2 else n2) / 3

pml_layers = [mp.PML(pml_thickness)]

chi_rad = math.radians(chi_deg)

zeta_rad = math.radians(zeta_deg)
kt = n1 * k_vac * math.cos(zeta_rad)
kl = n1 * k_vac * math.sin(zeta_rad)


cell = mp.Vector3(sx,sy,sz)
default_material = mp.Medium(index=n1)
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, sz),
                     center=mp.Vector3(0, 0, sz/(2 * math.cos(chi_rad))),
                     e1=mp.Vector3(math.cos(chi_rad), 0, math.sin(chi_rad)),
                     e2=mp.Vector3(0, 1, 0),
                     e3=mp.Vector3(-math.sin(chi_rad), 0, math.cos(chi_rad)),
                     material=mp.Medium(index=n2))]

print("\nExpected output file size:",
          round(8*(sx*sy*sz*resolution**3)/(1024**2)), "MiB")
          
force_complex_fields = True           # default: True
eps_averaging = True                  # default: True

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

for beam_type, amplitude in beams_and_amps:
    if amplitude == 0:
        continue
    sources.extend(make_beam_sources(beam_type(**all_beam_args), amplitude))

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    symmetries=[],
                    default_material=default_material,
                    Courant=Courant,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    force_complex_fields=force_complex_fields,
                    eps_averaging=eps_averaging
                    )

sim.use_output_directory()   # put output files in a separate folder

def efield_real_squared(r, ex, ey, ez):
    """Calculate |Re E|^2."""
    return ex.real**2 + ey.real**2 + ez.real**2

def efield_imag_squared(r, ex, ey, ez):
    """Calculate |Im E|^2."""
    return ex.imag**2 + ey.imag**2 + ez.imag**2

def output_efield_real_squared(sim):
    """Output E-field (real part) intensity."""
    name = "e_real2"
    func = efield_real_squared
    cs = [mp.Ex, mp.Ey, mp.Ez]
    return sim.output_field_function(name, cs, func, real_only=True)

def output_efield_imag_squared(sim):
    """Output E-field (imag part) intensity."""
    name = "e_imag2"
    func = efield_imag_squared
    cs = [mp.Ex, mp.Ey, mp.Ez]
    return sim.output_field_function(name, cs, func, real_only=True)

run_args = [#mp.at_beginning(mp.output_epsilon),    # output of dielectric function
            mp.at_end(mp.output_efield_x),         # output of E_x component
            mp.at_end(mp.output_efield_y),         # output of E_y component
            mp.at_end(mp.output_efield_z),         # output of E_z component
            mp.at_end(output_efield_real_squared),  # output of electric field intensity (real)
            mp.at_end(output_efield_imag_squared),   # output of electric field intensity (imag)
            # mp.at_every(0.2, mp.in_volume(mp.Volume(size=mp.Vector3(sx, sy, 0), center=mp.Vector3(0, 0, 2)),
            #                               mp.output_png(mp.Ex, "-Zc dkbluered"))),
            ]

sim.run(*run_args, until=runtime)

print("\nend time:", datetime.now())

