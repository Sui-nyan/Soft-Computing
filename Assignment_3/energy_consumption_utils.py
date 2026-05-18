"""
Energy Cost Utilities
This module calculates electricity costs using Taipei's tiered pricing structure.
"""

from typing import List, Tuple, Dict
import numpy as np


class TaipeiTieredPricing:
    """
    Calculates energy costs based on Taipei's tiered electricity rate structure.
    
    Pricing tiers (summer rates, in TWD per kWh):
    - 0-120 kWh:      1.78 TWD/kWh
    - 121-330 kWh:    2.55 TWD/kWh
    - 331-500 kWh:    3.80 TWD/kWh
    - 501-700 kWh:    4.80 TWD/kWh
    - 701+ kWh:       5.83 TWD/kWh
    """
    
    # Tier boundaries (cumulative kWh thresholds)
    TIER_BOUNDARIES = [120, 330, 500, 700, float('inf')]
    
    # Rate per kWh for each tier (TWD)
    TIER_RATES = [1.78, 2.55, 3.80, 4.80, 5.83]
    
    def __init__(self, season: str = 'summer'):
        """
        Initialize the pricing calculator.
        
        Parameters:
        -----------
        season : str
            Season for rate selection ('summer' or 'winter').
            Currently only summer rates are implemented.
        """
        self.season = season
        if season != 'summer':
            raise NotImplementedError("Only summer rates are currently implemented")
    
    def calculate_cost(self, total_consumption_kwh: float) -> float:
        """
        Calculate total electricity cost given daily consumption.
        
        Uses tiered pricing where each kWh in a higher tier is charged at 
        the higher rate (progressive billing).
        
        Parameters:
        -----------
        total_consumption_kwh : float
            Total daily energy consumption in kWh
        
        Returns:
        --------
        float
            Total cost in TWD (Taiwan Dollar)
        
        Mathematical formula:
        r(E) = {
            1.78 * E                                           if E ≤ 120
            1.78 * 120 + 2.55 * (E - 120)                     if 120 < E ≤ 330
            1.78 * 120 + 2.55 * 210 + 3.80 * (E - 330)       if 330 < E ≤ 500
            ... (continues for higher tiers)
        }
        """
        if total_consumption_kwh < 0:
            raise ValueError("Energy consumption cannot be negative")
        
        cost = 0.0
        remaining_kwh = total_consumption_kwh
        
        for tier_idx, (boundary, rate) in enumerate(zip(self.TIER_BOUNDARIES, self.TIER_RATES)):
            if remaining_kwh <= 0:
                break
            
            # Determine how much of this tier to charge
            if tier_idx == 0:
                tier_start = 0
            else:
                tier_start = self.TIER_BOUNDARIES[tier_idx - 1]
            
            # Amount of consumption in this tier
            tier_amount = min(remaining_kwh, boundary - tier_start)
            
            # Add cost for this tier
            cost += tier_amount * rate
            remaining_kwh -= tier_amount
        
        return cost
    
    def get_cost_breakdown(self, total_consumption_kwh: float) -> Dict[str, float]:
        """
        Get detailed cost breakdown by tier.
        
        Parameters:
        -----------
        total_consumption_kwh : float
            Total daily energy consumption in kWh
        
        Returns:
        --------
        dict
            Dictionary with keys:
            - 'tier_1_kwh': kWh consumed in tier 1
            - 'tier_1_cost': Cost for tier 1
            - 'tier_2_kwh': kWh consumed in tier 2
            - 'tier_2_cost': Cost for tier 2
            - ... (continues for all tiers)
            - 'total_kwh': Total consumption
            - 'total_cost': Total cost in TWD
            - 'effective_rate': Average cost per kWh
        """
        breakdown = {}
        cost = 0.0
        remaining_kwh = total_consumption_kwh
        
        for tier_idx, (boundary, rate) in enumerate(zip(self.TIER_BOUNDARIES, self.TIER_RATES)):
            if remaining_kwh <= 0:
                tier_amount = 0
                tier_cost = 0
            else:
                # Determine tier boundaries
                if tier_idx == 0:
                    tier_start = 0
                else:
                    tier_start = self.TIER_BOUNDARIES[tier_idx - 1]
                
                # Amount in this tier
                tier_amount = min(remaining_kwh, boundary - tier_start)
                tier_cost = tier_amount * rate
                
                remaining_kwh -= tier_amount
            
            tier_num = tier_idx + 1
            breakdown[f'tier_{tier_num}_kwh'] = tier_amount
            breakdown[f'tier_{tier_num}_cost'] = tier_cost
            cost += tier_cost
        
        breakdown['total_kwh'] = total_consumption_kwh
        breakdown['total_cost'] = cost
        
        # Effective rate (average cost per kWh)
        if total_consumption_kwh > 0:
            breakdown['effective_rate'] = cost / total_consumption_kwh
        else:
            breakdown['effective_rate'] = 0.0
        
        return breakdown
    
    def get_marginal_rate(self, total_consumption_kwh: float) -> float:
        """
        Get the marginal rate (cost per additional kWh at current consumption level).
        
        Parameters:
        -----------
        total_consumption_kwh : float
            Current total daily consumption in kWh
        
        Returns:
        --------
        float
            Marginal cost per additional kWh (TWD/kWh)
        """
        for boundary, rate in zip(self.TIER_BOUNDARIES, self.TIER_RATES):
            if total_consumption_kwh < boundary:
                return rate
        
        # Should never reach here for valid inputs
        return self.TIER_RATES[-1]
    
    def estimate_daily_cost_from_modes(
        self,
        modes: np.ndarray,
        ac_power: float = 2.9
    ) -> Tuple[float, float]:
        """
        Estimate daily cost directly from A/C mode schedule.
        
        NOTE: This is a simplified calculation that doesn't account for variable COP.
        For accurate results, use ThermalSimulator.simulate_day() instead.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of A/C modes for each hour (0 or 1)
        ac_power : float
            A/C power consumption in kWh/hour (default: 2.9)
        
        Returns:
        --------
        Tuple[float, float]
            - total_consumption: Total daily consumption in kWh
            - total_cost: Total daily cost in TWD
        """
        total_consumption = np.sum(modes) * ac_power
        total_cost = self.calculate_cost(total_consumption)
        return total_consumption, total_cost
    
    def calculate_cost_from_energy_kwh(self, energy_kwh: float) -> float:
        """
        Simple wrapper to calculate cost from energy in kWh.
        
        Parameters:
        -----------
        energy_kwh : float
            Total energy consumption in kWh
        
        Returns:
        --------
        float
            Cost in TWD
        """
        return self.calculate_cost(energy_kwh)


class CostAnalyzer:
    """
    Analyzes electricity costs and provides optimization insights.
    """
    
    def __init__(self):
        """Initialize the cost analyzer."""
        self.pricing = TaipeiTieredPricing(season='summer')
    
    def analyze_schedule(
        self,
        modes: np.ndarray,
        setpoints: np.ndarray,
        outdoor_temps: np.ndarray,
        ac_power: float = 2.9
    ) -> Dict:
        """
        Comprehensive cost and energy analysis for a given schedule.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of A/C modes for each hour
        setpoints : np.ndarray
            Array of temperature setpoints for each hour
        outdoor_temps : np.ndarray
            Array of outdoor temperatures for each hour
        ac_power : float
            A/C power consumption in kWh/hour
        
        Returns:
        --------
        dict
            Comprehensive analysis including:
            - 'total_consumption_kwh': Daily energy consumption
            - 'total_cost_twd': Daily electricity cost
            - 'effective_rate_twd_per_kwh': Average cost per kWh
            - 'marginal_rate_twd_per_kwh': Cost of next additional kWh
            - 'ac_on_hours': Number of hours A/C was ON
            - 'cost_breakdown': Tier-by-tier cost breakdown
        """
        total_consumption, total_cost = self.pricing.estimate_daily_cost_from_modes(
            modes, ac_power
        )
        marginal_rate = self.pricing.get_marginal_rate(total_consumption)
        cost_breakdown = self.pricing.get_cost_breakdown(total_consumption)
        
        ac_on_hours = int(np.sum(modes))
        
        return {
            'total_consumption_kwh': total_consumption,
            'total_cost_twd': total_cost,
            'effective_rate_twd_per_kwh': cost_breakdown['effective_rate'],
            'marginal_rate_twd_per_kwh': marginal_rate,
            'ac_on_hours': ac_on_hours,
            'cost_breakdown': cost_breakdown,
        }
    
    def compare_schedules(
        self,
        schedules: List[Tuple[np.ndarray, str]],
        outdoor_temps: np.ndarray,
        ac_power: float = 2.9
    ) -> Dict:
        """
        Compare cost metrics across multiple schedules.
        
        Parameters:
        -----------
        schedules : List[Tuple[np.ndarray, str]]
            List of (modes_array, schedule_name) tuples
        outdoor_temps : np.ndarray
            Array of outdoor temperatures for each hour
        ac_power : float
            A/C power consumption in kWh/hour
        
        Returns:
        --------
        dict
            Comparison results for each schedule
        """
        results = {}
        
        for modes, name in schedules:
            total_consumption, total_cost = self.pricing.estimate_daily_cost_from_modes(
                modes, ac_power
            )
            results[name] = {
                'consumption_kwh': total_consumption,
                'cost_twd': total_cost,
                'ac_on_hours': int(np.sum(modes)),
            }
        
        return results