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
from typing import Dict
from dataclasses import dataclass


@dataclass
class ThermalProperties:
    """
    Physical properties of the room and HVAC system.

    Assumptions: Taipei apartment, concrete construction, moderate insulation.
    Room dimensions: 2.5m height x 15m2 floor area = ~90m2 total surface.
    """

    # Thermal mass and resistance.
    # For the first-order model:
    #   dT/dt = (T_out - T_in)/(R*C) - Q_AC/C
    #
    # Units:
    #   R: K/W
    #   C: kJ/K
    #
    # Thermal capacitance realistic value for furnished room:
    # - Air: ~60 kJ/K
    # - Walls, furniture, thermal mass: ~200-300 kJ/K
    # - Total: ~400 kJ/K gives RC time constant of ~5.5 hours
    #
    # NOTE: Previous value of 50,000 kJ/K was too high (694 hour RC constant),
    # causing unrealistic room behavior where temp barely changes over a day.
    thermal_capacitance_kj_per_k: float = 400.0
    thermal_resistance_k_per_w: float = 0.05

    # A/C system parameters
    ac_max_power_w: float = 2900.0                  # Maximum cooling capacity (2.9 kW)
    ac_delta_t_ref_k: float = 5.0                   # Full power if room is 5C above setpoint

    # COP (Coefficient of Performance) parameters
    cop_rated: float = 3.5                          # COP at rated/ideal conditions
    cop_optimal_load_fraction: float = 0.70         # Load fraction where COP peaks
    cop_min: float = 2.2                            # Minimum COP at extreme loads

    # Comfort constraints
    acceptable_temp_min_c: float = 18.0             # Absolute minimum acceptable
    acceptable_temp_max_c: float = 28.0             # Absolute maximum acceptable

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
    - Low load: inefficient operation
    - Medium load (~70%): optimal operation
    - High load: less efficient operation

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
        self._fit_cop_curve()

    def _fit_cop_curve(self):
        """
        Fit a quadratic polynomial to COP as a function of load fraction.

        Constraints:
        - COP(0) = cop_min
        - COP(optimal) = cop_rated
        - COP(1) = cop_min

        Results in: COP(x) = a*x^2 + b*x + c
        """
        opt = self.props.cop_optimal_load_fraction
        rated = self.props.cop_rated
        min_cop = self.props.cop_min

        c = min_cop
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
            Fraction of maximum A/C capacity, from 0.0 to 1.0.

        Returns:
        --------
        float
            COP value. This method returns a valid curve value even at zero
            load, but the simulator records COP as 0.0 for zero-cooling hours
            so those hours are not included in mean operating COP.
        """
        load_fraction = np.clip(load_fraction, 0.0, 1.0)
        cop = self.a * load_fraction**2 + self.b * load_fraction + self.c
        return float(np.clip(cop, self.props.cop_min, self.props.cop_rated))

    def get_cop_array(self, load_fractions: np.ndarray) -> np.ndarray:
        """
        Vectorized COP calculation for arrays of load fractions.

        Parameters:
        -----------
        load_fractions : np.ndarray
            Array of load fractions.

        Returns:
        --------
        np.ndarray
            Array of corresponding COP values.
        """
        return np.array([self.get_cop(lf) for lf in load_fractions])


class ThermalSimulator:
    """
    Simulates hourly room temperature evolution with A/C operation.

    Uses a discrete hourly first-order thermal model:

        T_in(h+1) = T_in(h)
                    + (dt / RC) * (T_out(h) - T_in(h))
                    - (Q_AC(h) * dt) / C

    Where:
    - dt is the time step
    - RC is the thermal time constant
    - Q_AC is active A/C cooling power
    - C is thermal capacitance
    """

    def __init__(self, props: ThermalProperties = None):
        """
        Initialize thermal simulator.

        Parameters:
        -----------
        props : ThermalProperties
            Thermal properties. Uses defaults if None.
        """
        self.props = props if props is not None else ThermalProperties()
        self.cop_calc = COPCalculator(self.props)

        # Convert capacitance to SI units once.
        self.thermal_capacitance_j_per_k = self.props.thermal_capacitance_kj_per_k * 1000.0

        # R*C has units of seconds because:
        #   (K/W) * (J/K) = J/W = seconds
        self.rc_time_constant_seconds = (
            self.props.thermal_resistance_k_per_w * self.thermal_capacitance_j_per_k
        )
        self.rc_time_constant_hours = self.rc_time_constant_seconds / 3600.0

        # Backward-compatible attribute name used by get_properties_summary.
        # This is now explicitly in hours.
        self.rc_time_constant = self.rc_time_constant_hours

    def calculate_cooling_power(
        self,
        mode: int,
        indoor_temp: float,
        setpoint: float
    ) -> float:
        """
        Calculate A/C cooling power for this hour.

        Power scales with how far the room is from the setpoint:
        - If room <= setpoint: no cooling, thermostat satisfied
        - If room is 5C above setpoint: full power, 2.9 kW by default
        - If room is 1C above setpoint: 20% power, about 0.58 kW by default

        Parameters:
        -----------
        mode : int
            A/C mode, 0 = OFF and 1 = ON.
        indoor_temp : float
            Current indoor temperature in C.
        setpoint : float
            Desired temperature setpoint in C.

        Returns:
        --------
        float
            Cooling power in Watts.
        """
        if mode == 0:
            return 0.0

        temp_above_setpoint = max(0.0, indoor_temp - setpoint)
        load_fraction = min(1.0, temp_above_setpoint / self.props.ac_delta_t_ref_k)
        cooling_power = load_fraction * self.props.ac_max_power_w

        return float(cooling_power)

    def calculate_electrical_power(self, cooling_power: float) -> float:
        """
        Calculate electrical power consumed for a given cooling power.

        Electrical power = Cooling power / COP.

        Parameters:
        -----------
        cooling_power : float
            Cooling power in Watts.

        Returns:
        --------
        float
            Electrical power consumed in Watts.
        """
        if cooling_power <= 0:
            return 0.0

        load_fraction = cooling_power / self.props.ac_max_power_w
        cop = self.cop_calc.get_cop(load_fraction)
        electrical_power = cooling_power / cop

        return float(electrical_power)

    def update_temperature(
        self,
        current_temp: float,
        outdoor_temp: float,
        cooling_power: float,
        time_step_hours: float = 1.0
    ) -> float:
        """
        Update room temperature for one time step.

        Parameters:
        -----------
        current_temp : float
            Current room temperature in C.
        outdoor_temp : float
            Outdoor temperature during this period in C.
        cooling_power : float
            A/C cooling power in Watts.
        time_step_hours : float
            Time step duration in hours. Default is 1 hour.

        Returns:
        --------
        float
            New room temperature in C.
        """
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive")

        # Passive heat exchange: drives room toward outdoor temperature.
        passive_heat_change = (
            (outdoor_temp - current_temp) / self.rc_time_constant_hours
        ) * time_step_hours

        # Active cooling: removes heat from the indoor thermal mass.
        cooling_time_step_j = cooling_power * time_step_hours * 3600.0
        active_cooling_change = cooling_time_step_j / self.thermal_capacitance_j_per_k

        new_temp = current_temp + passive_heat_change - active_cooling_change

        return float(new_temp)

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
            A/C on/off command for each hour, shape (24,).
        setpoints : np.ndarray
            Temperature setpoints for each hour, shape (24,).
        outdoor_temps : np.ndarray
            Outdoor temperatures for each hour, shape (24,).
        initial_indoor_temp : float
            Starting indoor temperature in C. Default is 24C.

        Returns:
        --------
        dict
            Simulation results including:
            - indoor_temps: hourly indoor temperatures
            - cooling_powers: hourly cooling power in W
            - electrical_powers: hourly electrical power in W
            - cop_values: hourly COP values, 0.0 when no cooling occurs
            - total_energy_kwh: total electrical energy consumed
            - total_cooling_kwh: total cooling energy delivered
            - total_discomfort: sum of discomfort penalties
            - ac_on_hours: number of hours A/C command was ON
            - active_cooling_hours: number of hours with cooling_power > 0
        """
        modes = np.asarray(modes)
        setpoints = np.asarray(setpoints, dtype=float)
        outdoor_temps = np.asarray(outdoor_temps, dtype=float)

        if not (len(modes) == len(setpoints) == len(outdoor_temps)):
            raise ValueError("modes, setpoints, and outdoor_temps must have the same length")

        num_hours = len(modes)

        indoor_temps = np.zeros(num_hours)
        cooling_powers = np.zeros(num_hours)
        electrical_powers = np.zeros(num_hours)
        cop_values = np.zeros(num_hours)
        discomfort_values = np.zeros(num_hours)

        current_temp = float(initial_indoor_temp)

        for h in range(num_hours):
            cooling_power = self.calculate_cooling_power(
                mode=int(modes[h]),
                indoor_temp=current_temp,
                setpoint=setpoints[h]
            )

            electrical_power = self.calculate_electrical_power(cooling_power)

            if cooling_power > 0:
                load_fraction = cooling_power / self.props.ac_max_power_w
                cop = self.cop_calc.get_cop(load_fraction)
            else:
                cop = 0.0

            current_temp = self.update_temperature(
                current_temp=current_temp,
                outdoor_temp=outdoor_temps[h],
                cooling_power=cooling_power,
                time_step_hours=1.0
            )

            discomfort = self._calculate_discomfort(current_temp)

            indoor_temps[h] = current_temp
            cooling_powers[h] = cooling_power
            electrical_powers[h] = electrical_power
            cop_values[h] = cop
            discomfort_values[h] = discomfort

        # Each hourly power value contributes power * 1 hour.
        total_energy_kwh = np.sum(electrical_powers) / 1000.0
        total_cooling_kwh = np.sum(cooling_powers) / 1000.0
        total_discomfort = np.sum(discomfort_values)

        ac_on_hours = int(np.sum(modes.astype(int)))
        active_cooling_hours = int(np.sum(cooling_powers > 0))
        active_cop_values = cop_values[cooling_powers > 0]
        mean_cop = float(np.mean(active_cop_values)) if active_cop_values.size > 0 else 0.0

        return {
            'indoor_temps': indoor_temps,
            'cooling_powers': cooling_powers,
            'electrical_powers': electrical_powers,
            'cop_values': cop_values,
            'discomfort': discomfort_values,
            'total_energy_kwh': float(total_energy_kwh),
            'total_cooling_kwh': float(total_cooling_kwh),
            'total_discomfort': float(total_discomfort),
            'ac_on_hours': ac_on_hours,
            'active_cooling_hours': active_cooling_hours,
            'mean_indoor_temp': float(np.mean(indoor_temps)),
            'min_indoor_temp': float(np.min(indoor_temps)),
            'max_indoor_temp': float(np.max(indoor_temps)),
            'mean_cop': mean_cop,
        }

    def _calculate_discomfort(self, indoor_temp: float) -> float:
        """
        Calculate discomfort penalty (quadratic outside acceptable range).

        Parameters:
        -----------
        indoor_temp : float
            Current indoor temperature (C)

        Returns:
        --------
        float
            Discomfort penalty (C^2)
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
            Summary of key parameters.
        """
        return {
            'thermal_capacitance_kj_per_k': self.props.thermal_capacitance_kj_per_k,
            'thermal_resistance_k_per_w': self.props.thermal_resistance_k_per_w,
            'thermal_capacitance_j_per_k': self.thermal_capacitance_j_per_k,
            'rc_time_constant_seconds': self.rc_time_constant_seconds,
            'rc_time_constant_hours': self.rc_time_constant_hours,
            'ac_max_power_w': self.props.ac_max_power_w,
            'ac_delta_t_ref_k': self.props.ac_delta_t_ref_k,
            'cop_rated': self.props.cop_rated,
            'cop_optimal_load_fraction': self.props.cop_optimal_load_fraction,
            'cop_min': self.props.cop_min,
            'acceptable_temp_min_c': self.props.acceptable_temp_min_c,
            'acceptable_temp_max_c': self.props.acceptable_temp_max_c,
        }