Excellent choices! Variable cooling and discrete hourly updates give us a nice balance of realism and computational simplicity. Let me build out a thermodynamic model for you.

First, let me explain the physics I'm implementing, then show you the code:

---

## **The Discrete Hourly Thermal Model**

For each hour, the room temperature evolves as:

$$T_{in,h+1} = T_{in,h} + \frac{\Delta t}{RC} (T_{out,h} - T_{in,h}) + \frac{Q_{AC,h} \cdot \Delta t}{C}$$

Where:
- **T_in, h** = indoor temperature at hour h
- **T_out, h** = outdoor temperature during hour h  
- **RC** = thermal time constant (hours) — how fast the room exchanges heat with the environment
- **Q_AC, h** = cooling power from A/C (W) — **variable based on ΔT**
- **Δt** = 1 hour
- **C** = thermal capacitance (J/K)

**Key insight:** The term $(T_{out} - T_{in})/RC$ models passive heat gain/loss through walls. The term $Q_{AC}/C$ is active cooling.

---

## **Variable Cooling Power (Temperature-Dependent)**

Instead of constant 2.9 kW, we'll use:

$$Q_{AC,h} = \begin{cases}
0 & \text{if } m_h = 0 \text{ (A/C OFF)} \\
Q_{max} \cdot \min(1.0, \frac{T_{in,h} - T_{set,h}}{\Delta T_{ref}}) & \text{if } m_h = 1 \text{ (A/C ON)}
\end{cases}$$

Where:
- **Q_max** = maximum cooling power (e.g., 2.9 kW)
- **ΔT_ref** = reference temperature difference for full cooling (e.g., 5°C)

This means:
- If room is 5°C above setpoint → full cooling (2.9 kW)
- If room is 1°C above setpoint → ~20% cooling (0.58 kW)
- If room ≤ setpoint → no cooling (smart thermostat behavior)

---

## **Taipei Apartment Assumptions**

For a concrete high-rise apartment (typical in Taiwan):

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **C** (thermal mass) | ~50 kJ/K | Concrete walls, furniture, air |
| **R** (resistance) | ~0.05 K/W | Modern building code insulation |
| **RC time constant** | ~2.8 hours | Concrete buildings have significant thermal inertia |
| **Surface area** | 15 m² × 6 = 90 m² | 2.5m height × floor + ceiling + walls |
| **Air changes/hour** | ~0.5 | Sealed modern building (vs. 2-3 for older buildings) |

Before I write the code, **does this make intuitive sense to you?** 

A 2.8-hour time constant means:
- If A/C is OFF and outdoor temp is 35°C while room is 24°C, the room will drift toward 35°C, but it takes ~2.8 hours to go halfway there
- With A/C ON cooling hard, you pull temp down faster, but not *instantly*

Does that align with how you experience real A/C in Taipei?