# Point Matching Method of Moments (PM_MoM)

## Overview
This repository contains the implementation of the Point Matching Method of Moments for microstrip line analysis. The Point Matching method uses rectangular pulse basis functions with point testing to solve electromagnetic problems.

## Features
- **Point Matching Implementation**: Uses point testing with rectangular pulse basis functions
- **Convergence Analysis**: Detailed analysis of convergence with increasing N values
- **Empirical Comparison**: Comparison with empirical formulas from Notaros textbook
- **Surface Charge Distribution**: Visualization of charge distribution patterns
- **High-Resolution Analysis**: Non-uniform segmentation for better accuracy

## Files
- `PM_MoM.py`: Main implementation of the Point Matching method
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Results Summary

### Performance
- **Excellent convergence** with errors typically 0.03% - 3.23%
- **Proper convergence behavior** as N increases
- **Matches empirical formulas** very well

### Key Features
1. **Surface Charge Distribution Analysis** - Plots for different w/h ratios and N values
2. **Convergence Analysis** - Detailed comparison with empirical formulas
3. **High-Resolution Current Distribution** - Non-uniform segmentation analysis
4. **Edge Effects Analysis** - Charge concentration at strip edges

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python PM_MoM.py
```

## Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- tabulate >= 0.9.0

## Generated Outputs
- Surface charge distribution plots
- Convergence analysis plots
- High-resolution current distribution plots
- Edge effects analysis plots
- Comprehensive data tables

## Method Details

### Point Matching Method
The Point Matching method uses:
- **Basis Functions**: Rectangular pulses
- **Testing**: Point matching at segment centers
- **Impedance Matrix**: Z_ij = -(1/(2π*ε₀)) * [K1(δ/2, xi-xj, 0) - K1(δ/2, xi-xj, d)]

Where:
- K1 is the single integral over the logarithmic kernel
- δ is the segment width
- xi, xj are segment positions
- d = 2h is the distance parameter

## Results
The Point Matching method shows excellent performance with:
- Low errors (typically < 5%)
- Good convergence behavior
- Accurate results compared to empirical formulas

## Author
Alex Benson - ECE541 Applied Electromagnetics Project 1
