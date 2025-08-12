#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCA TS Optimization Plotter
Creates line plots of energy, gradients, and displacements from ORCA log files.

Usage: python plot_orca_optimization.py /path/to/temp/orca/folder/
"""

import os
import sys
import re
import numpy as np

def find_orca_log(folder):
    """Find the ORCA log file in the specified folder."""
    possible_names = ['orca_full.log', 'input.out', 'orca.out']
    
    for name in possible_names:
        log_path = os.path.join(folder, name)
        if os.path.isfile(log_path):
            return log_path
    
    return None

def parse_orca_log(log_path):
    """Parse ORCA log file to extract optimization data."""
    print(f"Parsing ORCA log file: {log_path}")
    
    cycles = []
    energies = []
    energy_changes = []
    rms_grads = []
    max_grads = []
    rms_disps = []
    max_disps = []
    
    # Thresholds from ORCA input (same as monitoring script)
    thresholds = {
        'energy_change': 1e-6,
        'rms_grad': 3e-4,
        'max_grad': 4.5e-4,
        'rms_disp': 1.2e-3,
        'max_disp': 1.8e-3
    }
    
    current_cycle = None
    current_data = {}
    previous_energy = None
    
    # Regex patterns (same as monitoring script)
    cycle_pattern = re.compile(r'ORCA GEOMETRY RELAXATION STEP')
    energy_pattern = re.compile(r'Current Energy\s+\.\.\.\.\s+([-+]?\d+\.\d+)\s+Eh')
    rms_grad_pattern = re.compile(r'RMS gradient\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)')
    max_grad_pattern = re.compile(r'MAX gradient\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)')
    rms_disp_pattern = re.compile(r'RMS step\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)')
    max_disp_pattern = re.compile(r'MAX step\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)')
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Detect new geometry optimization cycle
                if cycle_pattern.search(line):
                    if current_cycle is None:
                        current_cycle = 1
                    else:
                        current_cycle += 1
                    current_data = {'cycle': current_cycle}
                    continue
                
                # Extract current energy
                energy_match = energy_pattern.search(line)
                if energy_match and current_cycle is not None:
                    energy = float(energy_match.group(1))
                    current_data['energy'] = energy
                    
                    # Calculate energy change
                    if previous_energy is not None:
                        current_data['energy_change'] = energy - previous_energy
                    else:
                        current_data['energy_change'] = 0.0
                    
                    previous_energy = energy
                    continue
                
                # Parse convergence table values
                rms_grad_match = rms_grad_pattern.search(line)
                if rms_grad_match and current_cycle is not None:
                    current_data['rms_grad'] = float(rms_grad_match.group(1))
                    continue
                
                max_grad_match = max_grad_pattern.search(line)
                if max_grad_match and current_cycle is not None:
                    current_data['max_grad'] = float(max_grad_match.group(1))
                    continue
                
                rms_disp_match = rms_disp_pattern.search(line)
                if rms_disp_match and current_cycle is not None:
                    current_data['rms_disp'] = float(rms_disp_match.group(1))
                    continue
                
                max_disp_match = max_disp_pattern.search(line)
                if max_disp_match and current_cycle is not None:
                    current_data['max_disp'] = float(max_disp_match.group(1))
                    
                    # Check if we have complete data for this cycle
                    required_keys = ['cycle', 'energy', 'energy_change', 'rms_grad', 
                                   'max_grad', 'rms_disp', 'max_disp']
                    
                    if all(key in current_data for key in required_keys):
                        cycles.append(current_data['cycle'])
                        energies.append(current_data['energy'])
                        energy_changes.append(current_data['energy_change'])
                        rms_grads.append(current_data['rms_grad'])
                        max_grads.append(current_data['max_grad'])
                        rms_disps.append(current_data['rms_disp'])
                        max_disps.append(current_data['max_disp'])
                    
                    continue
    
    except Exception as e:
        print(f"Error parsing log file: {e}")
        sys.exit(1)
    
    print(f"Found {len(cycles)} optimization cycles")
    
    if len(cycles) == 0:
        print("No optimization data found in log file!")
        sys.exit(1)
    
    return {
        'cycles': np.array(cycles),
        'energies': np.array(energies),
        'energy_changes': np.array(energy_changes),
        'rms_grads': np.array(rms_grads),
        'max_grads': np.array(max_grads),
        'rms_disps': np.array(rms_disps),
        'max_disps': np.array(max_disps),
        'thresholds': thresholds
    }

def plot_matplotlib(data):
    """Create plots using matplotlib and save as PNG in current directory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install it with 'pip install matplotlib'")
        sys.exit(1)
    
    # Extract data
    cycles = data['cycles']
    energies = data['energies']
    rms_grads = data['rms_grads']
    max_grads = data['max_grads']
    rms_disps = data['rms_disps']
    max_disps = data['max_disps']
    thresholds = data['thresholds']
    
    # Create figure with custom height ratios
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                        gridspec_kw={'height_ratios': [2, 1, 1]},
                                        sharex=True)
    
    # Energy plot (top, double height)
    ax1.plot(cycles, energies, 'b-o', linewidth=2, markersize=4, label='Energy')
    ax1.set_ylabel('Energy (Ha)', fontsize=12, fontweight='bold')
    ax1.set_title('ORCA TS Optimization Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Format energy axis for better readability
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    # Gradient plot (middle)
    # Color points based on convergence thresholds
    rms_colors = ['green' if val < thresholds['rms_grad'] else 'red' for val in rms_grads]
    max_colors = ['green' if val < thresholds['max_grad'] else 'red' for val in max_grads]
    
    ax2.plot(cycles, rms_grads, 'o-', color='blue', alpha=0.7, linewidth=1, 
             markersize=3, label='RMS Grad')
    ax2.plot(cycles, max_grads, 's-', color='red', alpha=0.7, linewidth=1, 
             markersize=3, label='Max Grad')
    
    # Overlay colored points for threshold visualization
    ax2.scatter(cycles, rms_grads, c=rms_colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.scatter(cycles, max_grads, c=max_colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.5, marker='s')
    
    # Add threshold lines
    ax2.axhline(y=thresholds['rms_grad'], color='blue', linestyle='--', alpha=0.5, label=f'RMS threshold ({thresholds["rms_grad"]:.1e})')
    ax2.axhline(y=thresholds['max_grad'], color='red', linestyle='--', alpha=0.5, label=f'Max threshold ({thresholds["max_grad"]:.1e})')
    
    ax2.set_ylabel('Gradient (Eh/bohr)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    
    # Displacement plot (bottom)
    # Color points based on convergence thresholds
    rms_disp_colors = ['green' if val < thresholds['rms_disp'] else 'red' for val in rms_disps]
    max_disp_colors = ['green' if val < thresholds['max_disp'] else 'red' for val in max_disps]
    
    ax3.plot(cycles, rms_disps, 'o-', color='purple', alpha=0.7, linewidth=1, 
             markersize=3, label='RMS Disp')
    ax3.plot(cycles, max_disps, 's-', color='orange', alpha=0.7, linewidth=1, 
             markersize=3, label='Max Disp')
    
    # Overlay colored points for threshold visualization
    ax3.scatter(cycles, rms_disps, c=rms_disp_colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax3.scatter(cycles, max_disps, c=max_disp_colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.5, marker='s')
    
    # Add threshold lines
    ax3.axhline(y=thresholds['rms_disp'], color='purple', linestyle='--', alpha=0.5, label=f'RMS threshold ({thresholds["rms_disp"]:.1e})')
    ax3.axhline(y=thresholds['max_disp'], color='orange', linestyle='--', alpha=0.5, label=f'Max threshold ({thresholds["max_disp"]:.1e})')
    
    ax3.set_xlabel('Optimization Cycle', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Displacement (bohr)', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # FIXED: Save in current working directory instead of temp folder
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, "orca_TS_optimization.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"âœ… Plot saved as: {output_file}")
    
    # Also save a high-res version
    output_file_hires = os.path.join(current_dir, "orca_TS_optimization_hires.png")
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"âœ… High-res plot saved as: {output_file_hires}")
    
    plt.close()

def plot_plotext(data):
    """Create plots using plotext for terminal display with clean, simple visualization."""
    try:
        import plotext as plt
    except ImportError:
        print("plotext not installed. Install it with 'pip install plotext'")
        sys.exit(1)
    
    # Extract data
    cycles = data['cycles'].tolist()
    energies = data['energies'].tolist()
    rms_grads = data['rms_grads'].tolist()
    max_grads = data['max_grads'].tolist()
    rms_disps = data['rms_disps'].tolist()
    max_disps = data['max_disps'].tolist()
    thresholds = data['thresholds']

    
    # 1. Energy plot - simple and clean)
    
    plt.clear_figure()
    plt.plot(cycles, energies, marker="hd", color="blue")
    plt.title("Energy vs Optimization Cycle")
    plt.xlabel("Cycle")
    plt.ylabel("Energy (Ha)")
    plt.plotsize(80, 20)
    plt.show()
    
    # 2. Gradient plot - FIXED threshold lines
    print()
    plt.clear_figure()
    plt.plot(cycles, rms_grads, marker="hd", color="green", label="RMS Gradient")
    plt.plot(cycles, max_grads, marker="hd", color="red", label="Max Gradient")
    
    # FIXED: Remove label parameter from hline calls
    plt.hline(thresholds['rms_grad'], color="green")
    plt.hline(thresholds['max_grad'], color="red")
    
    plt.yscale("log")
    plt.title("Gradients vs Optimization Cycle (log scale)")
    plt.xlabel("Cycle")
    plt.ylabel("Gradient (Eh/bohr)")
    plt.plotsize(80, 20)
    plt.show()
    
    # 3. Displacement plot - FIXED threshold lines
    print()
    plt.clear_figure()
    plt.plot(cycles, rms_disps, marker="hd", color="magenta", label="RMS Displacement")
    plt.plot(cycles, max_disps, marker="hd", color="yellow", label="Max Displacement")
    
    # FIXED: Remove label parameter from hline calls
    plt.hline(thresholds['rms_disp'], color="magenta")
    plt.hline(thresholds['max_disp'], color="yellow")
    
    plt.yscale("log")
    plt.title("Displacements vs Optimization Cycle (log scale)")
    plt.xlabel("Cycle")
    plt.ylabel("Displacement (bohr)")
    plt.plotsize(80, 20)
    plt.show()
    
    # Simple convergence summary
    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60)
    
    latest_cycle = len(cycles)
    if latest_cycle > 0:
        latest_rms_grad = rms_grads[-1]
        latest_max_grad = max_grads[-1]
        latest_rms_disp = rms_disps[-1]
        latest_max_disp = max_disps[-1]
        
        print(f"Latest cycle: {latest_cycle}")
        print(f"Energy: {energies[-1]:.8f} Ha")
        print()
        print(f"RMS Gradient:     {latest_rms_grad:.6f} {'?' if latest_rms_grad < thresholds['rms_grad'] else '?'} (threshold: {thresholds['rms_grad']:.1e})")
        print(f"Max Gradient:     {latest_max_grad:.6f} {'?' if latest_max_grad < thresholds['max_grad'] else '?'} (threshold: {thresholds['max_grad']:.1e})")
        print(f"RMS Displacement: {latest_rms_disp:.6f} {'?' if latest_rms_disp < thresholds['rms_disp'] else '?'} (threshold: {thresholds['rms_disp']:.1e})")
        print(f"Max Displacement: {latest_max_disp:.6f} {'?' if latest_max_disp < thresholds['max_disp'] else '?'} (threshold: {thresholds['max_disp']:.1e})")
        
        converged = all([
            latest_rms_grad < thresholds['rms_grad'],
            latest_max_grad < thresholds['max_grad'],
            latest_rms_disp < thresholds['rms_disp'],
            latest_max_disp < thresholds['max_disp']
        ])
        
        print(f"\nOverall status: {'?? CONVERGED' if converged else '?? NOT CONVERGED'}")




def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python plot_orca_optimization.py /path/to/temp/orca/folder/")
        print("\nExample: python plot_orca_optimization.py /tmp/ExtOpt_AIQM2_LogMonitor_abc123/")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist")
        sys.exit(1)
    
    # Find ORCA log file
    log_file = find_orca_log(folder_path)
    if log_file is None:
        print(f"Error: No ORCA log file found in '{folder_path}'")
        print("Looking for: orca_full.log, input.out, or orca.out")
        sys.exit(1)
    
    # Parse the log file
    data = parse_orca_log(log_file)
    
    # Ask user for plotting backend
    print("\n" + "="*50)
    print("ORCA TS Optimization Plotter")
    print("="*50)
    print("Select plotting backend:")
    print("1) matplotlib (save high-quality PNG files)")
    print("2) plotext (display in terminal)")
    
    while True:
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == '1':
                plot_matplotlib(data)
                break
            elif choice == '2':
                plot_plotext(data)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)

if __name__ == "__main__":
    main()

