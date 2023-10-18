class Cryogen:
    """ Class which contains a cryogen thermodynamic
    and thermophysical properties """

    R = 8.314  # Ideal gas constant

    def __init__(
        self,
        name="EmptyCryogen",
        P=0,
        T_sat=0,
        rho_L=0,
        rho_V=0,
        h_L=0,
        h_V=0,
        k_V=0,
        cp_V=0,
        MW=0,
    ):
        """Constructor"""
        self.name = name
        self.P = P  # Pressure / Pa
        self.T_sat = T_sat  # Saturation temperature / K
        self.rho_L = rho_L  # Liquid Density / mol*m^-3
        self.rho_V = rho_V  # Vapour density / mol*m^-3
        self.rho_V_sat = rho_V  # Initialize vapour density at the interface
        self.h_L = h_L  # Liquid enthalpy J/mol
        self.h_V = h_V  # Vapour enthalpy J/mol
        self.k_V = k_V  # Thermal conductivity of the vapour W/mK
        self.k_int = k_V  # Thermal conductivity at the vapour-liquid interface
        self.cp_V = cp_V  # Heat capacity at constant pressure / J/molK
        self.MW = MW

    def rho_ig(self, T=None, P=None):
        """Returns ideal gas density in kg/m^3"""
        if P is None:  # Don't ask for P on isobaric conditions
            P = self.P
        if T is None:
            T = self.T_sat  # If T is not provided, use saturation temperature
        # Convert R (J mol^-1 K^-1) to R_gas (J kg^-1 K^-1)
        r_gas_si = self.R / self.MW * 1e3
        # Element-wise division work for numpy arrays
        rho_ig = P / (r_gas_si * T)
        return rho_ig
