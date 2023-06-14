from abc import ABC, abstractmethod
import cmath
from dataclasses import dataclass, field
import math

import meep as mp
from scipy.special import jv

class Beam3D(ABC):
    """Abstract class for a 3D beam
    
    Beam profile functions must usable as amp_function in mp.Source
    """

    @abstractmethod
    def x(self, r: mp.Vector3) -> complex:
        """Beam profile function for x component
        See meep documentation for more about beam profile functions:
        https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
        
        Args:
            r: Position relative to center of beam
        Returns:
            Complex amplitude at that point
        """
        pass

    @abstractmethod
    def y(self, r: mp.Vector3) -> complex:
        """Beam profile function for y component"""
        pass

    @abstractmethod
    def z(self, r: mp.Vector3) -> complex:
        """Beam profile function for z component"""
        pass

# Data class to autogenerate constructor
@dataclass
class BesselBeam3D(Beam3D):
    n1: float
    k_vac: float
    zeta_rad: float
    m_charge: int

    # Computed fields
    k: float = field(init=False)
    kt: float = field(init=False)
    kl: float = field(init=False)

    def __post_init__(self):
        self.k = self.n1 * self.k_vac
        self.kt = self.k * math.cos(self.zeta_rad)
        self.kl = self.k * math.sin(self.zeta_rad)

    # def __init__(self, n1: float, k_vac: float, zeta_rad: float) -> None:
    #     self.n1 = n1
    #     self.k_vac = k_vac
    #     self.zeta_rad = zeta_rad

    #     self.k = n1 * k_vac
    #     self.kt = self.k * math.cos(self.zeta_rad)
    #     self.kl = self.k * math.sin(self.zeta_rad)

    def f(self, n: float, r: mp.Vector3):
        phi = math.atan2(r.y, r.x)
        rho = math.sqrt(r.x**2 + r.y**2)
        return jv(n, self.kt * rho) * cmath.exp(1j * n * phi) * cmath.exp(1j * self.kl * r.z)
    
class TEBessel(BesselBeam3D):
    def x(self, r):
        return -0.5 * (self.kt/self.k) * (super().f(self.m_charge-1,r) + super().f(self.m_charge+1,r))

    def y(self, r):
        return 1j * 0.5 * (self.kt/self.k) * (super().f(self.m_charge+1,r) - super().f(self.m_charge-1,r))

    def z(self, r):
        return 0

class TMBessel(BesselBeam3D):
    def x(self, r):
        return 1j * 0.5 * (self.kt*self.kl/self.k**2) * (super().f(self.m_charge-1,r) - super().f(self.m_charge+1,r))

    def y(self, r):
        return -0.5 * (self.kt*self.kl/self.k**2) * (super().f(self.m_charge-1,r) + super().f(self.m_charge+1,r))

    def z(self, r):
        return (self.kt/self.k)**2 * super().f(self.m_charge,r)

class LEBessel(BesselBeam3D):
    def x(self, r):
        return (self.kl/self.k) * super().f(self.m_charge,r)

    def y(self, r):
        return 0
    
    def z(self, r):
        return 0.5 * 1j * (self.kt/self.k) * (super().f(self.m_charge-1,r) - super().f(self.m_charge+1,r))
    
class LMBessel(BesselBeam3D):
    def x(self, r):
        return 0.25 * 1j * (self.kt**2 / self.k**2) * (super().f(self.m_charge-2,r) - super().f(self.m_charge+2,r))

    def y(self, r):
        return (1/self.k**2) * ( 0.5 * (self.k**2 + self.kl**2) * super().f(self.m_charge,r) - 0.25 * self.kt**2 * (super().f(self.m_charge-2,r) + super().f(self.m_charge+2,r)) )

    def z(self, r):
        return -0.5 * self.kt * (self.kl/self.k**2) * (super().f(self.m_charge-1,r)+super().f(self.m_charge+1,r))

class CS1Bessel(BesselBeam3D):
    """CS1 Bessel Beam
    """
    def x(self, r):
        return (0.25/self.k**2) * ((self.k+self.kl)**2 * super().f(self.m_charge,r) + 0.5*self.kt**2 * (super().f(self.m_charge-2,r) + super().f(self.m_charge+2,r)))

    def y(self, r):
        return 0.125 * 1j * (self.kt/self.k)**2 * (super().f(self.m_charge-2,r) - super().f(self.m_charge+2,r))

    def z(self, r):
        return 0.25 * 1j * (self.kl + self.k) * (self.kt/self.k**2) * (super().f(self.m_charge-1,r) - super().f(self.m_charge+1,r))
    
class CS2Bessel(BesselBeam3D):
    def x(self, r):
        return (0.25/self.k**2) * ((self.k-self.kl)**2 * super().f(self.m_charge,r) + 0.5*self.kt**2 * (super().f(self.m_charge-2,r) + super().f(self.m_charge+2,r)))

    def y(self, r):
        return 0.125 * 1j * (self.kt/self.k)**2 * (super().f(self.m_charge-2,r) - super().f(self.m_charge+2,r))

    def z(self, r):
        return 0.25 * 1j * (self.kl + self.k) * (self.kt/self.k**2) * (super().f(self.m_charge-1,r) - super().f(self.m_charge+1,r))
