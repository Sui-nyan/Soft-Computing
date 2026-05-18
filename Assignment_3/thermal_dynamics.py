"""
Thermal Dynamics Module
Implements discrete hourly thermal simulation with variable COP efficiency.

Physical Model:
- First-order lumped thermal model for room temperature evolution
- Variable cooling power based on temperature difference
- COP (Coefficient of Performance) varies with operating load
- Accounts for passive heat exchange with environment
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class ThermalProperties:
    """
    Physical properties of the room and HVAC system.
    
    Assumptions: Taipei apartment, concrete construction, moderate insulation
    Room dimensions: 2.5m height × 15m² floor area = ~90m² total surface
    """
    
    # Thermal mass and resistance (for 1st-order model: dT/dt = (T_out - T_in)/(R*C) + Q_AC/C)
    thermal_capacitance_kj_per_k = 50.0      # Concrete walls + furniture + air (~50 MJ/K for this room)
    thermal_resistance_k_per_w = 0.05        # Insulation level (K/W) - moderate
    
    # A/C system parameters
    ac_max_power_w = 2900.0                  # Maximum cooling capacity (2.9 kW)
    ac_delta_t_ref_k = 5.0                   # Reference ΔT for full power (if room is 5°C above setpoint, run at full)
    
    # COP (Coefficient of Performance) parameters
    cop_rated = 3.5                          # COP at rated/ideal conditions
    cop_optimal_load_fraction = 0.70         # Load fraction (0-1) where COP peaks
    cop_min = 2.2                            # Minimum COP (at extreme loads)
    
    # Comfort constraints
    acceptable_temp_min_c = 18.0             # Absolute minimum acceptable
    acceptable_temp_max_c = 28.0             # Absolute maximum acceptable
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.thermal_capacitance_kj_per_k <= 0:
            raise ValueError("Thermal capacitance must be positive")
        if self.thermal_resistance_k_per_w <= 0:
            raise ValueError("Thermal resistance must be positive")
        if self.ac_max_power_w <= 0:
            raise ValueError("A/C power must be positive")
        if not (0 < self.cop_optimal_load_fraction < 1):
            raise ValueError("Optimal load fraction must be between 0 and 1")


class COPCalculator:
    """
    Calculates Coefficient of Performance (COP) as a function of operating load.
    
    COP varies because:
    - Low load: inefficient (compressor bypass, heat losses dominate)
    - Medium load (~70%): optimal (sweet spot of flow and pressure)
    - High load: less efficient (high pressure differences, heat losses)
    
    Uses a quadratic function peaking at optimal_load_fraction.
    """
    
    def __init__(self, props: ThermalProperties):
        """
        Initialize COP calculator.
        
        Parameters:
        -----------
        props : ThermalProperties
            Thermal properties object with COP parameters
        """
        self.props = props
        
        # Derive quadratic coefficients for COP(load_fraction)
        # We want: COP peaks at optimal_load_fraction with value cop_rated
        # And: COP = cop_min at load_fraction = 0 and load_fraction = 1
        self._fit_cop_curve()
    
    def _fit_cop_curve(self):
        """
        Fit a quadratic polynomial to COP as a function of load fraction.
        
        Constraints:
        - COP(0) = cop_min (no cooling = inefficient)
        - COP(optimal) = cop_rated (rated efficiency at sweet spot)
        - COP(1) = cop_min (maximum load = also inefficient)
        
        Results in: COP(x) = a*x² + b*x + c
        """
        opt = self.props.cop_optimal_load_fraction
        rated = self.props.cop_rated
        min_cop = self.props.cop_min
        
        # From constraint COP(0) = min_cop: c = min_cop
        c = min_cop
        
        # From constraint COP(1) = min_cop: a + b + c = min_cop
        # Therefore: a + b = 0, so b = -a
        
        # From constraint COP(opt) = rated:
        # a*opt² - a*opt + c = rated
        # a*opt(opt - 1) = rated - c
        a = (rated - c) / (opt * (opt - 1))
        b = -a
        
        self.a = a
        self.b = b
        self.c = c
    
    def get_cop(self, load_fraction: float) -> float:
        """
        Calculate COP for a given load fraction.
        
        Parameters:
        -----------
        load_fraction : float
            Fraction of maximum A/C capacity (0.0 to 1.0)
            - 0.0: A/C off
            - 0.5: 50% of max power
            - 1.0: 100% of max power
        
        Returns:
        --------
        float
            COP value (dimensionless)
        
        Mathematical formula:
        COP(load) = a*load² + b*load + c
        where coefficients are fit to peak at optimal load fraction
        """
        load_fraction = np.clip(load_fraction, 0.0, 1.0)
        
        cop = self.a * load_fraction**2 + self.b * load_fraction + self.c
        
        # Ensure COP stays within reasonable bounds
        return np.clip(cop, self.props.cop_min, self.props.cop_rated)
    
    def get_cop_array(self, load_fractions: np.ndarray) -> np.ndarray:
        """
        Vectorized COP calculation for arrays of load fractions.
        
        Parameters:
        -----------
        load_fractions : np.ndarray
            Array of load fractions
        
        Returns:
        --------
        np.ndarray
            Array of corresponding COP values
        """
        return np.array([self.get_cop(lf) for lf in load_fractions])


class ThermalSimulator:
    """
    Simulates hourly room temperature evolution with A/C operation.
    
    Uses discrete hourly first-order thermal model:
    T_in(h+1) = T_in(h) + (Δt/RC)*(T_out(h) - T_in(h)) + (Q_AC(h)*Δt)/C
    
    Where:
    - Δt = 1 hour
    - RC = thermal time constant (hours)
    - Q_AC = active A/C cooling (W)
    - C = thermal capacitance (J/K)
    """
    
    def __init__(self, props: ThermalProperties = None):
        """
        Initialize thermal simulator.
        
        Parameters:
        -----------
        props : ThermalProperties
            Thermal properties (uses defaults if None)
        """
        self.props = props if props is not None else ThermalProperties()
        self.cop_calc = COPCalculator(self.props)
        
        # Precompute thermal time constant for efficiency
        self.rc_time_constant = (
            self.props.thermal_resistance_k_per_w * 
            self.props.thermal_capacitance_kj_per_k
        )
    
    def calculate_cooling_power(
        self,
        mode: int,
        indoor_temp: float,
        setpoint: float
    ) -> float:
        """
        Calculate A/C cooling power for this hour.
        
        Power scales with how far room is from setpoint:
        - If room <= setpoint: no cooling (thermostat satisfied)
        - If room 5°C above setpoint: full power (2.9 kW)
        - If room 1°C above setpoint: 20% power (~0.58 kW)
        
        Parameters:
        -----------
        mode : int
            A/C mode (0 = OFF, 1 = ON)
        indoor_temp : float
            Current indoor temperature (°C)
        setpoint : float
            Desired temperature setpoint (°C)
        
        Returns:
        --------
        float
            Cooling power in Watts
        
        Mathematical formula:
        Q_AC = {
            0                                    if mode = 0
            Q_max * min(1, (T_in - T_set)/ΔT_ref)  if mode = 1 and T_in > T_set
            0                                    if mode = 1 and T_in ≤ T_set
        }
        """
        if mode == 0:
            return 0.0
        
        # Calculate how much cooling is needed
        temp_above_setpoint = max(0.0, indoor_temp - setpoint)
        
        # Fraction of maximum power (0.0 to 1.0)
        load_fraction = min(1.0, temp_above_setpoint / self.props.ac_delta_t_ref_k)
        
        # Actual cooling power
        cooling_power = load_fraction * self.props.ac_max_power_w
        
        return cooling_power
    
    def calculate_electrical_power(
        self,
        cooling_power: float
    ) -> float:
        """
        Calculate electrical power consumed given desired cooling power.
        
        Accounts for COP efficiency:
        Electrical power = Cooling power / COP
        
        Parameters:
        -----------
        cooling_power : float
            Cooling power desired (W)
        
        Returns:
        --------
        float
            Electrical power consumed (W)
        
        Mathematical formula:
        P_electrical = Q_AC / COP(load_fraction)
        where load_fraction = Q_AC / Q_max
        """
        if cooling_power == 0:
            return 0.0
        
        load_fraction = cooling_power / self.props.ac_max_power_w
        cop = self.cop_calc.get_cop(load_fraction)
        
        electrical_power = cooling_power / cop
        
        return electrical_power
    
    def update_temperature(
        self,
        current_temp: float,
        outdoor_temp: float,
        cooling_power: float,
        time_step_hours: float = 1.0
    ) -> float:
        """
        Update room temperature for one time step.
        
        Uses first-order thermal model:
        T_in(h+1) = T_in(h) + (Δt/RC)*(T_out - T_in) + (Q_AC*Δt)/C
        
        Where:
        - First term: passive heat exchange with environment
        - Second term: active cooling from A/C
        
        Parameters:
        -----------
        current_temp : float
            Current room temperature (°C)
        outdoor_temp : float
            Outdoor temperature during this period (°C)
        cooling_power : float
            A/C cooling power (W)
        time_step_hours : float
            Time step duration (default: 1 hour)
        
        Returns:
        --------
        float
            New room temperature (°C)
        """
        # Passive heat exchange: drives room toward outdoor temp
        # Rate depends on R, C and current temperature difference
        heat_exchange_rate = (outdoor_temp - current_temp) / self.rc_time_constant
        passive_heat_change = heat_exchange_rate * time_step_hours
        
        # Active cooling: removes heat at rate = cooling_power
        # Convert cooling_power (W = J/s) to temperature change (°C)
        cooling_time_step_j = cooling_power * time_step_hours * 3600  # W * hours * 3600 s/hour = J
        cooling_capacitance_j_per_k = self.props.thermal_capacitance_kj_per_k * 1000  # Convert to J/K
        active_cooling_change = cooling_time_step_j / cooling_capacitance_j_per_k
        
        # Net temperature change
        new_temp = current_temp + passive_heat_change - active_cooling_change
        
        return new_temp
    
    def simulate_day(
        self,
        modes: np.ndarray,
        setpoints: np.ndarray,
        outdoor_temps: np.ndarray,
        initial_indoor_temp: float = 24.0
    ) -> Dict:
        """
        Simulate a full day of A/C operation.
        
        Parameters:
        -----------
        modes : np.ndarray
            A/C on/off for each hour (shape: 24)
        setpoints : np.ndarray
            Temperature setpoints for each hour (shape: 24)
        outdoor_temps : np.ndarray
            Outdoor temperatures for each hour (shape: 24)
        initial_indoor_temp : float
            Starting indoor temperature (default: 24°C)
        
        Returns:
        --------
        dict
            Simulation results including:
            - 'indoor_temps': Hourly indoor temperatures
            - 'cooling_powers': Hourly cooling power (W)
            - 'electrical_powers': Hourly electrical power (W)
            - 'cop_values': Hourly COP values
            - 'total_energy_kwh': Total electrical energy consumed
            - 'total_cooling_kwh': Total cooling energy delivered
            - 'total_discomfort': Sum of discomfort penalties
            - 'ac_on_hours': Number of hours A/C was ON
        """
        num_hours = len(modes)
        
        # Initialize output arrays
        indoor_temps = np.zeros(num_hours)
        cooling_powers = np.zeros(num_hours)
        electrical_powers = np.zeros(num_hours)
        cop_values = np.zeros(num_hours)
        discomfort_values = np.zeros(num_hours)
        
        current_temp = initial_indoor_temp
        
        # Simulate each hour
        for h in range(num_hours):
            # Calculate cooling power needed
            cooling_power = self.calculate_cooling_power(
                mode=int(modes[h]),
                indoor_temp=current_temp,
                setpoint=setpoints[h]
            )
            
            # Calculate electrical power required
            electrical_power = self.calculate_electrical_power(cooling_power)
            
            # Get COP for this operating point
            load_fraction = cooling_power / self.props.ac_max_power_w if cooling_power > 0 else 0
            cop = self.cop_calc.get_cop(load_fraction)
            
            # Update indoor temperature
            current_temp = self.update_temperature(
                current_temp=current_temp,
                outdoor_temp=outdoor_temps[h],
                cooling_power=cooling_power,
                time_step_hours=1.0
            )
            
            # Calculate discomfort penalty
            discomfort = self._calculate_discomfort(current_temp)
            
            # Store results
            indoor_temps[h] = current_temp
            cooling_powers[h] = cooling_power
            electrical_powers[h] = electrical_power
            cop_values[h] = cop
            discomfort_values[h] = discomfort
        
        # Calculate aggregates
        total_energy_kwh = np.sum(electrical_powers) / 1000.0 / 3600.0  # W to kWh conversion (W * hours / 1e6)
        # Actually: electrical_powers is in W, each hour contributes electrical_power * 1 hour
        # So total energy = sum(electrical_powers) * 1 hour / 1000 = sum(electrical_powers) / 1000 kWh
        total_energy_kwh = np.sum(electrical_powers) * 1.0 / 1000.0  # W * hours / 1000 = kWh
        
        total_cooling_kwh = np.sum(cooling_powers) * 1.0 / 1000.0  # W * hours / 1000 = kWh
        total_discomfort = np.sum(discomfort_values)
        ac_on_hours = int(np.sum(modes))
        
        return {
            'indoor_temps': indoor_temps,
            'cooling_powers': cooling_powers,
            'electrical_powers': electrical_powers,
            'cop_values': cop_values,
            'discomfort': discomfort_values,
            'total_energy_kwh': total_energy_kwh,
            'total_cooling_kwh': total_cooling_kwh,
            'total_discomfort': total_discomfort,
            'ac_on_hours': ac_on_hours,
            'mean_indoor_temp': np.mean(indoor_temps),
            'min_indoor_temp': np.min(indoor_temps),
            'max_indoor_temp': np.max(indoor_temps),
            'mean_cop': np.mean(cop_values[cop_values > 0]) if np.any(cop_values > 0) else 0,
        }
    
    def _calculate_discomfort(self, indoor_temp: float) -> float:
        """
        Calculate discomfort penalty (quadratic outside acceptable range).
        
        Parameters:
        -----------
        indoor_temp : float
            Current indoor temperature (°C)
        
        Returns:
        --------
        float
            Discomfort penalty (°C²)
        """
        if indoor_temp > self.props.acceptable_temp_max_c:
            return (indoor_temp - self.props.acceptable_temp_max_c) ** 2
        elif indoor_temp < self.props.acceptable_temp_min_c:
            return (self.props.acceptable_temp_min_c - indoor_temp) ** 2
        else:
            return 0.0
    
    def get_properties_summary(self) -> Dict:
        """
        Get a summary of thermal properties for documentation.
        
        Returns:
        --------
        dict
            Summary of key parameters
        """
        return {
            'thermal_capacitance_kj_per_k': self.props.thermal_capacitance_kj_per_k,
            'thermal_resistance_k_per_w': self.props.thermal_resistance_k_per_w,
            'rc_time_constant_hours': self.rc_time_constant,
            'ac_max_power_w': self.props.ac_max_power_w,
            'cop_rated': self.props.cop_rated,
            'cop_optimal_load_fraction': self.props.cop_optimal_load_fraction,
            'cop_min': self.props.cop_min,
        }