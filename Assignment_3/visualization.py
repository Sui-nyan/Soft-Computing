import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ac_simulation(indoor_temps, ac_modes, outdoor_temps=None, title="A/C Simulation 28-Day Results", save_filename=None):
    """
    繪製室內外溫度變化與冷氣開關狀態的折線圖
    """
    hours = np.arange(len(indoor_temps))
    
    fig, ax1 = plt.subplots(figsize=(18, 6))

    
    # plot temperature curves (y-axis on the left)

    ax1.set_xlabel('Time (Hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', color='black', fontsize=12)
    
    ax1.plot(hours, indoor_temps, label='Indoor Temp', color='darkblue', linewidth=2.0)
    
    if outdoor_temps is not None:
        ax1.plot(hours, outdoor_temps, label='Outdoor Temp', color='darkorange', alpha=0.9, linewidth=1.5)
        
    ax1.axhline(y=28.0, color='red', linestyle='--', alpha=0.4, label='Comfort Max (28°C)')
    ax1.axhline(y=18.0, color='blue', linestyle='--', alpha=0.4, label='Comfort Min (18°C)')
    
    ax1.tick_params(axis='y', labelcolor='black')
 
    ax1.legend(loc='lower right',bbox_to_anchor=(1.0, -0.12), ncol=4, framealpha=1.0,facecolor='white',edgecolor='darkgray')
    ax1.grid(True, alpha=0.3)
    
    # 設定 X 軸刻度為每 24 小時一個標記 (代表一天)
    ax1.set_xticks(np.arange(0, len(hours)+1, 24))

    # ============================
    # 繪製冷氣狀態 (對應右側 Y 軸)
    # ============================
    ax2 = ax1.twinx()
    ax2.set_ylabel('A/C State (1=ON, 0=OFF)', color='tab:green', fontsize=12)
    
    ax2.fill_between(hours, 0, ac_modes, step="post", color='tab:green', alpha=0.2)
    ax2.step(hours, ac_modes, where="post", color='tab:green', alpha=0.8, linewidth=1.5)
    
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([0, 1])
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(title, fontsize=16, pad=20)
    fig.tight_layout()
 
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_convergence(csv_filename='convergence_analysis_summer.csv', title='Convergence Analysis', save_filename=None):
    """
    Read the convergence data CSV and plot the convergence trend of the genetic algorithm.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"⚠️ File not found: {csv_filename}. Please ensure the algorithm has successfully exported this file.")
        return

    plt.figure(figsize=(10, 6))

    scenarios = df['Scenario'].unique() 
    print(f"Plotting convergence chart for scenarios: {scenarios}")

    for scenario in scenarios:
        sub_df = df[df['Scenario'] == scenario]
        plt.plot(sub_df['Generation'], sub_df['Best Fitness'], label=scenario, linewidth=2)

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (Cost / Penalty)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(loc='upper right', framealpha=0.9) 

    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot successfully saved as {save_filename}")
        
    plt.show()