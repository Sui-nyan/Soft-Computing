"""
Energy Consumption Utilities
This module calculates indoor temperatures and energy consumption for A/C scheduling.
"""

import numpy as np
from typing import Tuple, List


class EnergyConsumptionCalculator:
    """
    Calculates energy-related metrics for A/C scheduling.
    
    Parameters:
    -----------
    AC_power : float
        A/C power consumption in kWh/hour when ON (default: 2.9)
    """
    
    def __init__(self, AC_power: float = 2.9):
        """Initialize the energy calculator with A/C power rating."""
        self.AC_power = AC_power
    
    def calculate_indoor_temperature(
        self,
        mode: int,
        setpoint: float,
        outdoor_temp: float,
        temp_drop_when_off: float = 2.0
    ) -> float:
        """
        Calculate indoor temperature for a given hour.
        
        Parameters:
        -----------
        mode : int
            A/C mode (0 = OFF, 1 = ON)
        setpoint : float
            Temperature setpoint (°C), only used if mode == 1
        outdoor_temp : float
            Outdoor temperature (°C)
        temp_drop_when_off : float
            Temperature drop from outdoor when A/C is OFF (default: 2.0°C)
        
        Returns:
        --------
        float
            Indoor temperature (°C)
        
        Mathematical formula:
        T_indoor,h = {
            T_set,h          if m_h = 1 (A/C ON)
            T_outdoor,h - 2  if m_h = 0 (A/C OFF)
        }
        """
        if mode == 1:
            return setpoint
        else:
            return outdoor_temp - temp_drop_when_off
    
    def calculate_hourly_consumption(self, mode: int) -> float:
        """
        Calculate energy consumption for a single hour.
        
        Parameters:
        -----------
        mode : int
            A/C mode (0 = OFF, 1 = ON)
        
        Returns:
        --------
        float
            Energy consumption in kWh
        
        Mathematical formula:
        E_h = P_AC * m_h = {
            2.9 kWh  if m_h = 1
            0 kWh    if m_h = 0
        }
        """
        return self.AC_power * mode
    
    def calculate_hourly_discomfort(self, indoor_temp: float) -> float:
        """
        Calculate discomfort penalty using quadratic penalty function.
        
        Parameters:
        -----------
        indoor_temp : float
            Indoor temperature (°C)
        
        Returns:
        --------
        float
            Discomfort penalty (°C²)
        
        Mathematical formula:
        D_h = {
            (T_indoor,h - 28)²  if T_indoor,h > 28
            (18 - T_indoor,h)²  if T_indoor,h < 18
            0                    if 18 ≤ T_indoor,h ≤ 28
        }
        """
        acceptable_min = 18
        acceptable_max = 28
        
        if indoor_temp > acceptable_max:
            return (indoor_temp - acceptable_max) ** 2
        elif indoor_temp < acceptable_min:
            return (acceptable_min - indoor_temp) ** 2
        else:
            return 0.0
    
    def calculate_daily_metrics(
        self,
        modes: np.ndarray,
        setpoints: np.ndarray,
        outdoor_temps: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Calculate all daily metrics from hourly decisions.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of A/C modes for each hour (shape: 24)
        setpoints : np.ndarray
            Array of temperature setpoints for each hour (shape: 24)
        outdoor_temps : np.ndarray
            Array of outdoor temperatures for each hour (shape: 24)
        
        Returns:
        --------
        Tuple[np.ndarray, float, float]
            - indoor_temps: Array of indoor temperatures for each hour
            - total_consumption: Total daily energy consumption (kWh)
            - total_discomfort: Total daily discomfort penalty (°C²)
        """
        num_hours = len(modes)
        
        # Initialize arrays
        indoor_temps = np.zeros(num_hours)
        hourly_consumption = np.zeros(num_hours)
        hourly_discomfort = np.zeros(num_hours)
        
        # Calculate for each hour
        for h in range(num_hours):
            # Indoor temperature
            indoor_temps[h] = self.calculate_indoor_temperature(
                mode=int(modes[h]),
                setpoint=setpoints[h],
                outdoor_temp=outdoor_temps[h]
            )
            
            # Hourly energy consumption
            hourly_consumption[h] = self.calculate_hourly_consumption(
                mode=int(modes[h])
            )
            
            # Hourly discomfort
            hourly_discomfort[h] = self.calculate_hourly_discomfort(
                indoor_temp=indoor_temps[h]
            )
        
        # Calculate totals
        total_consumption = np.sum(hourly_consumption)
        total_discomfort = np.sum(hourly_discomfort)
        
        return indoor_temps, total_consumption, total_discomfort
    
    def get_consumption_summary(
        self,
        modes: np.ndarray,
        setpoints: np.ndarray,
        outdoor_temps: np.ndarray
    ) -> dict:
        """
        Get a comprehensive summary of energy consumption and comfort metrics.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of A/C modes for each hour
        setpoints : np.ndarray
            Array of temperature setpoints for each hour
        outdoor_temps : np.ndarray
            Array of outdoor temperatures for each hour
        
        Returns:
        --------
        dict
            Dictionary with keys:
            - 'indoor_temps': Array of indoor temperatures
            - 'total_consumption_kwh': Total daily consumption
            - 'total_discomfort': Total daily discomfort penalty
            - 'ac_on_hours': Number of hours A/C was ON
            - 'mean_indoor_temp': Mean indoor temperature
            - 'min_indoor_temp': Minimum indoor temperature
            - 'max_indoor_temp': Maximum indoor temperature
        """
        indoor_temps, total_consumption, total_discomfort = self.calculate_daily_metrics(
            modes, setpoints, outdoor_temps
        )
        
        ac_on_hours = int(np.sum(modes))
        
        return {
            'indoor_temps': indoor_temps,
            'total_consumption_kwh': total_consumption,
            'total_discomfort': total_discomfort,
            'ac_on_hours': ac_on_hours,
            'mean_indoor_temp': np.mean(indoor_temps),
            'min_indoor_temp': np.min(indoor_temps),
            'max_indoor_temp': np.max(indoor_temps),
        }