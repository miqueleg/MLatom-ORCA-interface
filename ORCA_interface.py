#!/usr/bin/env python3

import os, sys, json, shutil, subprocess, tempfile, re, time
import numpy as np
import mlatom as ml
from pathlib import Path
from datetime import datetime
import threading

# =================== USER CONFIGURATION ===================
ORCA = "/home/mestevez/Programs/Orca/orca"  # Path to ORCA executable
THREADS = 16                     # Cores for AIQM2 calculations
CHARGE = 0                       # Molecular charge
MULT = 1                         # Spin multiplicity
FREEZE = [1,9,10,19,23,24,33,44,45,55,58,59,64,66,67,72,83,84,94,103,104,113,121,122,136,150,160,163,166,177,187,196,206,217,218,259,272,276,281,286,287,299,300,303]  # Atoms to freeze (1-based)
FIX_DISTANCES = []              # List of distance constraints: [(atom1, atom2, distance_angstrom), ...] e.g., [(2, 3, 1.5)]
SCAN_DISTANCES = [[232,229,3.14255,1.45498,25]]              # List of distance to SCAN if OPT_TYPE == "SCAN": [(atom1, atom2, minimum_distance, maximum_distance, Number_of_Interpolations), ...] e.g., [(2, 3, 1.5, 2.4, 20)]
XTBKW = "--etemp 1000.0 --iterations 1000 --albp ether"  # xTB keywords for AIQM2
MAX_STEPS = 1000                  # Maximum optimization steps
INITIAL_HESSIAN = False          # Compute initial Hessian (expensive but accurate)
HESSIAN_RECALC_FREQ = 0        # Recalculate exact Hessian every X cycles (0 = disable)
OPT_TYPE = "SCAN"                  # Optimization type: "TS" for transition state (OptTS), "MIN" for minimization (Opt), "SCAN" for Scan
# ===========================================================

# Enhanced ANSI colors for better visibility
ANSI = {
    "green": "\033[92m",     # Bright green for converged values
    "cyan": "\033[96m",      # Cyan for headers
    "yellow": "\033[93m",    # Yellow for warnings/info
    "red": "\033[91m",       # Red for errors
    "blue": "\033[94m",      # Blue for titles
    "magenta": "\033[95m",   # Magenta for emphasis
    "bold": "\033[1m",       # Bold text
    "end": "\033[0m"         # Reset
}

class ORCALogMonitor:
    """Background monitor that reads ORCA log file and updates console table."""
    
    def __init__(self, log_file_path, update_interval=0.5):
        self.log_file_path = log_file_path
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.last_position = 0
        self.cycles_printed = set()
        self.header_printed = False  # Track if header has been printed
        
        # Track current cycle data
        self.current_cycle = None
        self.current_data = {}
        
        # Convergence thresholds
        self.thresholds = {
            'energy_change': 1e-6,
            'rms_grad': 3e-4,
            'max_grad': 4.5e-4,
            'rms_disp': 1.2e-3,
            'max_disp': 1.8e-3
        }
        
        # DON'T print header here anymore - wait until first data row
    
    def print_header(self):
        """Print the convergence table header."""
        if not self.header_printed:
            print(f"\n{'Cycle':>5} {'Energy (Ha)':>15}{'?E':>12}{'RMS Grad':>11}{'Max Grad':>11}{'RMS Disp':>11}{'Max Disp':>11}{'Status':>12}")
            print("-" * 89)
            self.header_printed = True
    
    def color_value(self, value, threshold, format_str="{:.2e}", width=11):
        """Color value green if below threshold with proper width formatting."""
        # Format the value first
        formatted = format_str.format(abs(value))
        
        # Apply color if below threshold
        if abs(value) < threshold:
            colored = f"{ANSI['green']}{formatted}{ANSI['end']}"
            # Calculate padding needed - the ANSI codes add 9 characters total (\033[92m = 5, \033[0m = 4)
            # But we want the same visual width as uncolored text
            visible_width = len(formatted)
            padding_needed = width - visible_width
            if padding_needed > 0:
                return " " * padding_needed + colored
            else:
                return colored
        else:
            # For uncolored text, use standard right-alignment
            return f"{formatted:>{width}}"
    
    def start_monitoring(self):
        """Start the background monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """Stop the background monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.running:
            try:
                if os.path.exists(self.log_file_path):
                    self._read_new_content()
                time.sleep(self.update_interval)
            except Exception as e:
                # Silent error handling - don't interrupt optimization
                pass
    
    def _read_new_content(self):
        """Read new content from log file since last check."""
        try:
            with open(self.log_file_path, 'r') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()
                
                if new_content:
                    self._process_content(new_content)
        except:
            pass  # File might be locked during writing
    
    def _process_content(self, content):
        """Process new log content and extract optimization data."""
        lines = content.split('\n')
        
        for line in lines:
            self._extract_data_from_line(line)
    
    def _extract_data_from_line(self, line):
        """Extract optimization data from ORCA log line."""
        
        # NEW: Detect Hessian recalculation
        if "HESSIAN RECALC" in line:
            hess_match = re.search(r'HESSIAN RECALC \[Cycle (\d+)\]: E=([-+]?\d+\.\d+) Ha, Imag=(\d+)', line)
            if hess_match:
                cycle = int(hess_match.group(1))
                energy = float(hess_match.group(2))
                n_imag = int(hess_match.group(3))
                
                if n_imag == 1:
                    hess_status = f"{ANSI['green']}Perfect TS{ANSI['end']}"
                elif n_imag > 1:
                    hess_status = f"{ANSI['yellow']}{n_imag} imaginary{ANSI['end']}"
                else:
                    hess_status = f"{ANSI['red']}No TS character{ANSI['end']}"
                
                print(f"    {ANSI['blue']}Hessian recalc: {hess_status} (E={energy:.6f}){ANSI['end']}")
            return
        
        # Detect new geometry optimization cycle
        if "ORCA GEOMETRY RELAXATION STEP" in line:
            # If we have incomplete previous cycle, don't start new one yet
            if self.current_cycle is not None and self.current_cycle not in self.cycles_printed:
                return
            
            # Find the actual cycle number (look ahead in parsing or use counter)
            if self.current_cycle is None:
                self.current_cycle = 1
            else:
                self.current_cycle += 1
            
            self.current_data = {'cycle': self.current_cycle}
            return
        
        # Extract current energy
        energy_match = re.search(r'Current Energy\s+\.\.\.\.\s+([-+]?\d+\.\d+)\s+Eh', line)
        if energy_match and self.current_cycle is not None:
            energy = float(energy_match.group(1))
            self.current_data['energy'] = energy
            
            # Calculate energy change if we have previous energy
            if hasattr(self, 'previous_energy'):
                self.current_data['energy_change'] = energy - self.previous_energy
            else:
                self.current_data['energy_change'] = 0.0
            
            self.previous_energy = energy
            return
        
        # Extract gradient norm (we'll use this as an indicator)
        grad_norm_match = re.search(r'Current gradient norm\s+\.\.\.\.\s+([-+]?\d+\.\d+)\s+Eh/bohr', line)
        if grad_norm_match and self.current_cycle is not None:
            grad_norm = float(grad_norm_match.group(1))
            self.current_data['grad_norm'] = grad_norm
            return
        
        # Parse the convergence table
        # RMS gradient        0.0182020702            0.0003000000      NO
        rms_grad_match = re.search(r'RMS gradient\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)', line)
        if rms_grad_match and self.current_cycle is not None:
            self.current_data['rms_grad'] = float(rms_grad_match.group(1))
            return
        
        # MAX gradient        0.5613339384            0.0004500000      NO
        max_grad_match = re.search(r'MAX gradient\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)', line)
        if max_grad_match and self.current_cycle is not None:
            self.current_data['max_grad'] = float(max_grad_match.group(1))
            return
        
        # RMS step            0.0022233204            0.0012000000      NO
        rms_step_match = re.search(r'RMS step\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)', line)
        if rms_step_match and self.current_cycle is not None:
            self.current_data['rms_disp'] = float(rms_step_match.group(1))
            return
        
        # MAX step            0.0302426067            0.0018000000      NO
        max_step_match = re.search(r'MAX step\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+(YES|NO)', line)
        if max_step_match and self.current_cycle is not None:
            self.current_data['max_disp'] = float(max_step_match.group(1))
            
            # Check if we have all required data to print the cycle
            self._check_and_print_cycle()
            return
        
        # Check for convergence messages
        if "The optimization has converged" in line or "HURRAY" in line:
            print(f"{ANSI['green']}OPTIMIZATION CONVERGED!{ANSI['end']}")
            return
        
        if "more geometry cycles are needed" in line:
            # This indicates end of current cycle - good place to ensure we print it
            self._check_and_print_cycle()
            return
    
    def _check_and_print_cycle(self):
        """Check if we have complete cycle data and print table row."""
        if (self.current_cycle is not None and 
            self.current_cycle not in self.cycles_printed and
            'energy' in self.current_data and
            'rms_grad' in self.current_data and
            'max_grad' in self.current_data and
            'rms_disp' in self.current_data and
            'max_disp' in self.current_data):
            
            # Print header ONLY before the first data row
            self.print_header()
            
            self._print_cycle_row(self.current_data)
            self.cycles_printed.add(self.current_cycle)
            
            # Update trajectory
            update_trajectory("input_EXT.xyz", self.current_cycle)
    
    def _print_cycle_row(self, data):
        """Print a single table row for a complete cycle."""
        
        # Check convergence
        criteria_met = [
            abs(data['energy_change']) < self.thresholds['energy_change'],
            data['rms_grad'] < self.thresholds['rms_grad'],
            data['max_grad'] < self.thresholds['max_grad'],
            data['rms_disp'] < self.thresholds['rms_disp'],
            data['max_disp'] < self.thresholds['max_disp']
        ]
        
        status = f"{ANSI['green']}CONVERGED{ANSI['end']}" if all(criteria_met) else "Running"
        
        # Format values with FIXED width formatting that accounts for ANSI codes
        energy_str = f"{data['energy']:15.8f}"
        delta_e_str = self.color_value(data['energy_change'], self.thresholds['energy_change'], width=12)
        rms_grad_str = self.color_value(data['rms_grad'], self.thresholds['rms_grad'], width=11)
        max_grad_str = self.color_value(data['max_grad'], self.thresholds['max_grad'], width=11)
        rms_disp_str = self.color_value(data['rms_disp'], self.thresholds['rms_disp'], width=11)
        max_disp_str = self.color_value(data['max_disp'], self.thresholds['max_disp'], width=11)
        
        # Format status with proper width
        status_formatted = f"{status:>22}" if "CONVERGED" in status else f"{status:>12}"
        
        # Print the table row - no additional formatting needed since widths are handled in color_value
        print(f"{data['cycle']:>5} {energy_str}{delta_e_str}{rms_grad_str}{max_grad_str}{rms_disp_str}{max_disp_str}{status_formatted}")

def create_xtb_keyword_file():
    """Create xTB keyword file for AIQM2."""
    with open("xtbkw", 'w') as f:
        f.write(XTBKW + "\n")

def create_aiqm2_method():
    """Create MLatom AIQM2 method instance."""
    return ml.models.methods(
        method="AIQM2",
        nthreads=THREADS,
        qm_program_kwargs={"read_keywords_from_file": "xtbkw"}
    )

def compute_initial_hessian(elements, coords):
    """Compute initial Hessian with AIQM2 if requested."""
    if not INITIAL_HESSIAN:
        return None
    
    print(f"{ANSI['yellow']}Computing initial AIQM2 Hessian ({len(elements)} atoms)...{ANSI['end']}")
    print(f"This will require ~{6 * len(elements)} gradient calculations and may take 10-30 minutes.")
    
    # Create XYZ string
    xyz_lines = [str(len(elements)), "Initial Hessian calculation"]
    for el, (x, y, z) in zip(elements, coords):
        xyz_lines.append(f"{el} {x:.10f} {y:.10f} {z:.10f}")
    mol_xyz = "\n".join(xyz_lines)
    
    # Create molecule and compute Hessian
    mol = ml.molecule.from_xyz_string(mol_xyz)
    method = create_aiqm2_method()
    
    start_time = time.time()
    method.predict(
        molecule=mol,
        calculate_energy=True,
        calculate_energy_gradients=True,
        calculate_hessian=True
    )
    elapsed = time.time() - start_time
    
    energy = float(mol.energy)
    hessian = np.array(mol.hessian)
    
    # Analyze Hessian for transition state character
    eigenvals = np.linalg.eigvals(hessian)
    n_imag = np.sum(eigenvals < -1e-6)
    
    print(f"{ANSI['green']}Initial Hessian computed in {elapsed:.1f} seconds{ANSI['end']}")
    print(f"Initial energy: {energy:.8f} Ha")
    print(f"Imaginary frequencies: {n_imag}")
    
    if n_imag == 1:
        print(f"{ANSI['green']}Good TS guess: exactly 1 imaginary frequency{ANSI['end']}")
    elif n_imag > 1:
        print(f"{ANSI['yellow']}Multiple imaginary frequencies ({n_imag}): structure may be unstable{ANSI['end']}")
    else:
        print(f"{ANSI['yellow']}No imaginary frequencies: this may not be a TS{ANSI['end']}")
    
    return hessian

def write_orca_input(elements, coords):
    """Generate ORCA input for serial ExtOpt."""
    
    # Determine optimization keyword based on OPT_TYPE
    if OPT_TYPE == "TS":
        opt_keyword = "OptTS TightSCF SlowConv"
    elif OPT_TYPE == "MIN":
        opt_keyword = "Opt"
    elif OPT_TYPE == "SCAN":
        opt_keyword = "Opt TightSCF"
        
    else:
        raise ValueError(f"Invalid OPT_TYPE: {OPT_TYPE}. Use 'TS', 'MIN' or 'SCAN'.")
    
    # Generate constraints block
    constraints = ["  Constraints"]
    
    # Add atom freezing constraints
    if FREEZE:
        for i in FREEZE:
            constraints.append(f"    {{C {i-1} C}}")  # ORCA uses 0-based indexing for constraints
    
    # Add distance fixing constraints
    for atom1, atom2, dist in FIX_DISTANCES:
        constraints.append(f"    {{B {atom1-1} {atom2-1} {dist:.4f} C}}")  # 0-based, distance in Angstroms
    
    constraints.append("  end")
    constraint_block = "\n" + "\n".join(constraints) if (FREEZE or FIX_DISTANCES) else ""
    
    # Generate Scan block
    constraints_scan = ["  Scan"]
    
    # Add distance fixing constraints
    for atom1, atom2, dist1, dist2, steps in SCAN_DISTANCES:
        constraints_scan.append(f"    B {atom1-1} {atom2-1} = {dist1:.4f}, {dist2:.4f}, {steps}")  # 0-based, distance in Angstroms
    
    constraints_scan.append("  end")
    Scan_block = "\n" + "\n".join(constraints_scan) if SCAN_DISTANCES else ""
    
    # Generate geometry section
    geom_section = "\n".join(f"{el:2s} {x:16.10f} {y:16.10f} {z:16.10f}"
                            for el, (x, y, z) in zip(elements, coords))
    
    input_text = f"""# Serial ORCA ExtOpt with Parallel AIQM2 Calculator
! ExtOpt {opt_keyword} NoUseSym

%method
ProgExt "./aiqm2_orca_wrapper.py"
end

%geom
    MaxIter {MAX_STEPS}
    Trust 0.1
    Calc_Hess false
    
    # Optimization convergence criteria
    TolE 1e-6
    TolRMSG 3e-4
    TolMaxG 4.5e-4
    TolRMSD 1.2e-3
    TolMaxD 1.8e-3{constraint_block}{Scan_block}
end

* xyz {CHARGE} {MULT}
{geom_section}
*
"""
    
    with open("input.inp", "w") as f:
        f.write(input_text)
    
    if FREEZE:
        print(f"Freezing {len(FREEZE)} atoms: {FREEZE}")
    if FIX_DISTANCES:
        print(f"Fixing {len(FIX_DISTANCES)} distances: {FIX_DISTANCES}")
    print(f"{ANSI['yellow']}Using serial ORCA ExtOpt mode with {opt_keyword}{ANSI['end']}")

def write_aiqm2_wrapper(elements):
    """Create AIQM2 wrapper script with periodic Hessian recalculation."""
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIQM2 wrapper for ORCA 6.1 ExtOpt
Handles energy and gradient calculations with periodic Hessian recalculation
"""

import sys, os, json, numpy as np, mlatom as ml
import traceback

def get_current_cycle():
    """Get current optimization cycle using a reliable counter file."""
    counter_file = "cycle_counter.txt"
    
    if not os.path.exists(counter_file):
        with open(counter_file, 'w') as f:
            f.write("1")
        return 1
    else:
        with open(counter_file, 'r') as f:
            cycle = int(f.read().strip())
        with open(counter_file, 'w') as f:
            f.write(str(cycle + 1))
        return cycle

def should_compute_hessian(cycle_num):
    """
    Decide whether an exact Hessian has to be recomputed.
    """
    hessian_freq = {HESSIAN_RECALC_FREQ}        # 0 = never
    initial_hessian = {INITIAL_HESSIAN}         # True / False

    if hessian_freq <= 0:
        return False            # user disabled periodic Hessians
    if cycle_num == 1 and initial_hessian:
        return False            # already provided at start
    return cycle_num > 0 and cycle_num % hessian_freq == 0


def compute_aiqm2_with_hessian(elements_list, coords, cycle_num):
    """Compute AIQM2 energy, gradient, and Hessian."""
    with open("calc_params.json", 'r') as f:
        params = json.load(f)
    
    with open("xtbkw", "w") as f:
        f.write(params.get("xtb_keywords_content", "{XTBKW}"))
    
    method = ml.models.methods(
        method="AIQM2",
        nthreads={THREADS},
        qm_program_kwargs={{"read_keywords_from_file": "xtbkw"}}
    )
    
    xyz_lines = [str(len(elements_list)), f"Hessian recalc cycle {{cycle_num}}"]
    for el, (x, y, z) in zip(elements_list, coords):
        xyz_lines.append(f"{{el}} {{x:.10f}} {{y:.10f}} {{z:.10f}}")
    mol_xyz = "\\n".join(xyz_lines)
    
    mol = ml.molecule.from_xyz_string(mol_xyz)
    method.predict(
        molecule=mol,
        calculate_energy=True,
        calculate_energy_gradients=True,
        calculate_hessian=True
    )
    
    energy = float(mol.energy)
    gradient = mol.energy_gradients.flatten()
    hessian = np.array(mol.hessian)
    
    eigenvals = np.linalg.eigvals(hessian)
    n_imag = np.sum(eigenvals < -1e-6)
    
    print(f"HESSIAN RECALC [Cycle {{cycle_num}}]: E={{energy:.8f}} Ha, Imag={{n_imag}}")
    
    hess_file = f"cycle_{{cycle_num}}_hessian.txt"
    with open(hess_file, 'w') as f:
        f.write(f"# Cycle {{cycle_num}} Hessian Analysis\\n")
        f.write(f"# Energy: {{energy:.12f}} Ha\\n") 
        f.write(f"# Imaginary frequencies: {{n_imag}}\\n")
        f.write(f"# Eigenvalues (first 10): {{eigenvals[:10]}}\\n")
        np.savetxt(f, hessian, fmt='%16.10e')
    
    return energy, gradient

def compute_aiqm2_regular(elements_list, coords):
    """Compute regular AIQM2 energy and gradient."""
    with open("calc_params.json", 'r') as f:
        params = json.load(f)
    
    with open("xtbkw", "w") as f:
        f.write(params.get("xtb_keywords_content", "{XTBKW}"))
    
    method = ml.models.methods(
        method="AIQM2",
        nthreads={THREADS},
        qm_program_kwargs={{"read_keywords_from_file": "xtbkw"}}
    )
    
    xyz_lines = [str(len(elements_list)), "AIQM2 calculation for ORCA"]
    for el, (x, y, z) in zip(elements_list, coords):
        xyz_lines.append(f"{{el}} {{x:.10f}} {{y:.10f}} {{z:.10f}}")
    mol_xyz = "\\n".join(xyz_lines)
    
    mol = ml.molecule.from_xyz_string(mol_xyz)
    method.predict(
        molecule=mol,
        calculate_energy=True,
        calculate_energy_gradients=True,
        calculate_hessian=False
    )
    
    energy = float(mol.energy)
    gradient = mol.energy_gradients.flatten()
    
    return energy, gradient

def main():
    try:
        # Parse command line arguments
        if len(sys.argv) != 2:
            print("Error: Expected 1 argument (input file)")
            sys.exit(1)
        
        inp_file = sys.argv[1]
        out_file = inp_file + ".out"
        
        # Parse ORCA ExtOpt input format
        with open(inp_file, 'r') as f:
            meta_lines = [line.strip() for line in f if line.strip()]
        
        # Extract XYZ filename from first line
        xyz_filename = meta_lines[0].split('#')[0].strip()
        
        # Read coordinates from XYZ file
        elements_list = []
        coords_list = []
        
        with open(xyz_filename, 'r') as f:
            n_atoms = int(f.readline().strip())
            f.readline()  # Skip comment line
            
            for _ in range(n_atoms):
                parts = f.readline().strip().split()
                elements_list.append(parts[0])
                coords_list.append([float(x) for x in parts[1:4]])
        
        coords = np.array(coords_list)
        
        # Get current cycle number
        cycle_num = get_current_cycle()
        
        # Check if we should compute Hessian
        if should_compute_hessian(cycle_num):
            energy, gradient = compute_aiqm2_with_hessian(elements_list, coords, cycle_num)
        else:
            energy, gradient = compute_aiqm2_regular(elements_list, coords)
        
        # Write .out file (ExtOpt format)
        with open(out_file, "w") as f:
            f.write("energy\\n")
            f.write(f"{{energy:.12f}}\\n")
            f.write("gradient\\n")
            
            # Write gradients in groups of 3 (x,y,z per atom)
            for i in range(0, len(gradient), 3):
                f.write(f"{{gradient[i]:16.10e}} {{gradient[i+1]:16.10e}} {{gradient[i+2]:16.10e}}\\n")
        
        # Write .engrad file (ORCA standard format)
        engrad_file = xyz_filename.replace('.xyz', '.engrad')
        with open(engrad_file, 'w') as f:
            f.write(f"# Number of atoms\\n{{len(elements_list)}}\\n")
            f.write(f"# The current total energy in Eh\\n{{energy:.12f}}\\n")
            f.write("# The current gradient in Eh/bohr\\n")
            
            # Convert gradients from Eh/Angstrom to Eh/bohr (one value per line)
            bohr_to_angstrom = 0.529177249
            for grad_component in (gradient / bohr_to_angstrom):
                f.write(f"{{grad_component:16.10e}}\\n")
        
        # Success message
        max_grad = np.max(np.abs(gradient))
        hess_status = " [HESS]" if should_compute_hessian(cycle_num - 1) else ""
        print(f"AIQM2{{hess_status}}: E={{energy:.8f}} Ha, max|grad|={{max_grad:.6f}} ({THREADS} cores)")
        
    except Exception as e:
        print(f"WRAPPER ERROR: {{str(e)}}")
        traceback.print_exc()
        
        # Write dummy results to prevent ORCA from hanging
        try:
            with open(out_file, "w") as f:
                f.write("energy\\n0.0\\ngradient\\n")
                for i in range(len(elements_list)):
                    f.write("0.0 0.0 0.0\\n")
            
            engrad_file = xyz_filename.replace('.xyz', '.engrad')
            with open(engrad_file, 'w') as f:
                f.write(f"# Error case\\n{{len(elements_list)}}\\n# Energy\\n0.0\\n# Gradient\\n")
                for _ in range(len(elements_list) * 3):
                    f.write("0.0\\n")
        except:
            pass
        
        sys.exit(1)
    
    finally:
        # Cleanup temporary files
        for temp_file in ["xtbkw"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

if __name__ == "__main__":
    main()
'''
    
    # Write wrapper script and make executable
    with open("aiqm2_orca_wrapper.py", "w") as f:
        f.write(script_content)
    os.chmod("aiqm2_orca_wrapper.py", 0o755)

def update_trajectory(xyz_file, cycle):
    """Silently update trajectory file."""
    if not os.path.exists(xyz_file):
        return
    
    try:
        with open("orca_traj.xyz", "a") as traj:
            with open(xyz_file, 'r') as src:
                lines = src.readlines()
                if len(lines) >= 2:
                    lines[1] = f"Cycle {cycle:3d} - {datetime.now().strftime('%H:%M:%S')}\n"
                    traj.writelines(lines)
    except:
        pass  # Silent failure - don't interrupt optimization

def main():
    """Main driver function."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"{ANSI['red']}Usage: python {sys.argv[0]} structure.xyz{ANSI['end']}")
        print(f"Example: python {sys.argv[0]} enzyme_ts_guess.xyz")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    
    if not os.path.exists(xyz_file):
        print(f"{ANSI['red']}Error: Input file '{xyz_file}' not found{ANSI['end']}")
        sys.exit(1)
    
    # Read input geometry
    elements = []
    coords = []
    
    try:
        with open(xyz_file, 'r') as f:
            n_atoms = int(f.readline().strip())
            f.readline()  # Skip comment line
            
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        elements.append(parts[0])
                        coords.append([float(x) for x in parts[1:4]])
    
        coords = np.array(coords)
        
    except Exception as e:
        print(f"{ANSI['red']}Error reading XYZ file: {e}{ANSI['end']}")
        sys.exit(1)
    
    # Display configuration
    opt_desc = "TS Optimization" if OPT_TYPE == "TS" else "Energy Minimization"
    print(f"{ANSI['cyan']}ORCA ExtOpt + AIQM2 {opt_desc} (Background Log Monitor){ANSI['end']}")
    print(f"Input file: {xyz_file}")
    print(f"Atoms: {len(elements)}")
    print(f"Frozen atoms: {len(FREEZE) if FREEZE else 0}")
    print(f"Fixed distances: {len(FIX_DISTANCES) if FIX_DISTANCES else 0}")
    print(f"ORCA mode: Serial (ExtOpt)")
    print(f"AIQM2 threads: {THREADS}")
    print(f"Charge: {CHARGE}, Multiplicity: {MULT}")
    print(f"Initial Hessian: {'Yes (expensive!)' if INITIAL_HESSIAN else 'No (BFGS updates only)'}")
    print(f"Hessian recalc: {'Every ' + str(HESSIAN_RECALC_FREQ) + ' cycles' if HESSIAN_RECALC_FREQ > 0 else 'Disabled'}")
    print(f"Max steps: {MAX_STEPS}")
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp(prefix="ExtOpt_AIQM2_LogMonitor_")
    print(f"Working directory: {temp_dir}")
    
    old_cwd = os.getcwd()
    monitor = None
    
    try:
        os.chdir(temp_dir)
        
        # Setup calculation environment
        create_xtb_keyword_file()
        
        # Save parameters for wrapper script
        calc_params = {
            "elements": elements,
            "xtb_keywords_content": XTBKW,
            "nthreads": THREADS
        }
        
        with open("calc_params.json", "w") as f:
            json.dump(calc_params, f)
        
        # Compute initial Hessian if requested
        hessian = compute_initial_hessian(elements, coords)
        
        # Generate ORCA input and wrapper script
        write_orca_input(elements, coords)
        write_aiqm2_wrapper(elements)
        
        # Initialize empty trajectory file
        Path("orca_traj.xyz").write_text("")
        
        # ========== CORRECTED LOG MONITORING ==========
        
        # Monitor the correct file: orca_full.log
        log_file_path = os.path.join(temp_dir, "orca_full.log")
        
        # Start background log monitor (header will be printed when first data appears)
        monitor = ORCALogMonitor(log_file_path, update_interval=0.5)
        monitor.start_monitoring()
        
        print(f"\n{ANSI['yellow']}Starting ORCA ExtOpt optimization...{ANSI['end']}")
        print(f"{ANSI['blue']}Background log monitoring active - table will update in real-time{ANSI['end']}")
        
        # Simple ORCA execution - let it run without stdout parsing
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['ORCA_NUM_PROCS'] = '1'
        
        # Run ORCA with line buffering for immediate output
        with open("orca_full.log", 'w', buffering=1) as log_file:
            process = subprocess.Popen(
                [ORCA, "input.inp"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            
            # Wait for completion while monitor runs in background
            return_code = process.wait()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Give monitor a moment to process final output
        time.sleep(2)
        
        # Check results
        if return_code == 0:
            print(f"\n{ANSI['green']}Optimization completed successfully!{ANSI['end']}")
        elif return_code == 1:
            print(f"\n{ANSI['yellow']}ORCA reached MaxIter ({MAX_STEPS}) without convergence{ANSI['end']}")
            print("This is normal for difficult cases - check trajectory for progress")
        else:
            print(f"\n{ANSI['red']}ORCA finished with return code {return_code}{ANSI['end']}")
            print("Check orca_full.log for detailed error information")
        
        
        print(f"\n{ANSI['cyan']}Results copied to: {old_cwd}{ANSI['end']}")
        shutil.copy2('orca_full.log', old_cwd)
        shutil.copy2('input_trj.xyz', old_cwd+'/orca_traj.xyz')
        shutil.copy2('input.xyz', old_cwd+'/output.xyz')
        print(f"--> orca_traj.xyz: Optimization trajectory")
        print(f"--> output.xyz: Final optimized geometry")
        print(f"--> orca_full.log: ORCA output file")
        
        # Final summary
        if os.path.exists("input_EXT.xyz"):
            # Count trajectory frames
            with open("input_EXT.xyz", 'r') as f:
                content = f.read()
                n_frames = content.count(f"{len(elements)}\n")
            
            print(f"\n{ANSI['cyan']}=== Optimization Summary ==={ANSI['end']}")
            print(f"Trajectory frames: {n_frames}")
            print(f"Timestamped trajectory: orca_traj.xyz")
            print(f"Full log: orca_full.log")
            print(f"Working directory: {temp_dir}")
        
    except KeyboardInterrupt:
        print(f"\n{ANSI['yellow']}Optimization interrupted by user (Ctrl+C){ANSI['end']}")
        if monitor:
            monitor.stop_monitoring()
        if 'process' in locals():
            process.terminate()
    
    except Exception as e:
        print(f"{ANSI['red']}Error during optimization: {e}{ANSI['end']}")
        import traceback
        traceback.print_exc()
        if monitor:
            monitor.stop_monitoring()
        
    finally:
        # Return to original directory
        os.chdir(old_cwd)
        
        # Keep temporary directory for debugging
        print(f"{ANSI['cyan']}Debug: Temporary directory preserved at {temp_dir}{ANSI['end']}")

if __name__ == "__main__":
    main()

