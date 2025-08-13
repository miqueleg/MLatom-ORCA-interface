# MLatom-ORCA-interface
My MLatom-ORCA interface for cluster-model calculations.
This is my implementation that I used for computing Opt, OptTS and SCAN calculations in ORCa, using MLatom as calculator.

OptTS calculations are not recomended because the hessian will be estimated by the optimization algorithm, and for complicated cases, this calculation will never complete.
To do so, I recomend using the MLatom-Sella interface: https://github.com/miqueleg/MLatom-Sella-TS-search

### Usage
- Install ORCA and MLatom and all the needed dependencies
- Download `ORCA-interface.py`
- Open the file and change the USER CONFIGURATION parameters to fit your calculation
- Execute `python Orca-interface.py input.xyz`

### Plots
Use `python ORCA_plot.py /path/to/temp/folder` to make plots to follow the progress of the calculation. (Requires matplotlib and plotext)
Select 2 to use plottext, and the plot will be printed in the terminal (perfect for tracking a remote calculation)
Select 1 for using matplotlib and get a .png  and a _hires.png files.

### Example:
SCAN:
<img width="2972" height="3570" alt="SCAN" src="https://github.com/user-attachments/assets/f0e3921b-e330-4826-af4b-132d44dae8fd" />
