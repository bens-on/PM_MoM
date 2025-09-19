import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def K1(a, x, y):
    """
    K1 function: negative integral from x'=-a to a of ln(sqrt((x-x')^2 + y^2)) dx'
    
    Parameters:
    a (float): half-width of the integration domain
    x (float): x-coordinate 
    y (float): y-coordinate
    
    Returns:
    float: negative integral value
    """
    def integrand(x_prime):
        """Integrand function for the K1 calculation"""
        distance_squared = (x - x_prime)**2 + y**2
        # Handle case where distance is very small to avoid log(0)
        if distance_squared < 1e-12:
            distance_squared = 1e-12
        return np.log(np.sqrt(distance_squared))
    
    # Perform numerical integration from -a to a
    integral_result, _ = quad(integrand, -a, a)
    
    # Return negative integral
    return -integral_result

def populate_impedance_matrix(N, w, h, x_positions):
    """
    Populate the impedance matrix Z using nested for loops
    
    Parameters:
    N (int): number of segments
    w (float): width of the microstrip
    h (float): height parameter
    x_positions (array): array of x positions for each segment
    
    Returns:
    numpy.ndarray: impedance matrix Z
    """
    # Constants
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    delta = w / N  # Width of each segment
    d = 2 * h  # d parameter
    
    # Initialize matrix
    Z = np.zeros((N, N))
    
    # Nested for loops to populate matrix
    for i in range(N):
        for j in range(N):
            xi = x_positions[i]
            xj = x_positions[j]
            
            # Calculate Z_ij using the formula
            term1 = K1(delta/2, xi - xj, 0)
            term2 = K1(delta/2, xi - xj, d)
            
            Z[i, j] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
    
    return Z

def calculate_segment_positions(N, w):
    """
    Calculate the x positions for each segment of the microstrip
    
    Parameters:
    N (int): number of segments
    w (float): width of the microstrip
    
    Returns:
    numpy.ndarray: array of x positions
    """
    # Create N segments across the width w
    # Position at the center of each segment
    delta = w / N
    x_positions = np.linspace(-w/2 + delta/2, w/2 - delta/2, N)
    
    return x_positions

def solve_point_match_system(Z, V):
    """
    Solve the linear system Z * I = V for current distribution
    
    Parameters:
    Z (numpy.ndarray): impedance matrix
    V (numpy.ndarray): voltage vector
    
    Returns:
    numpy.ndarray: current distribution I
    """
    # Solve the linear system
    I = np.linalg.solve(Z, V)
    return I

def calculate_surface_charge_density(I, delta):
    """
    Calculate surface charge density from current distribution
    
    Parameters:
    I (numpy.ndarray): current distribution
    delta (float): segment width
    
    Returns:
    numpy.ndarray: surface charge density
    """
    # Surface charge density is related to current density
    # For a microstrip, ρs = I / (segment_area)
    # Assuming unit length in z-direction, area = delta * 1
    rho_s = I / delta
    return rho_s

def empirical_capacitance_microstrip(w, h, epsilon_r=1.0):
    """
    Empirical formula for microstrip capacitance per unit length
    Based on Notaros textbook formulas
    
    Parameters:
    w (float): strip width
    h (float): substrate height
    epsilon_r (float): relative permittivity (default 1.0 for air)
    
    Returns:
    float: capacitance per unit length (F/m)
    """
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    
    # Effective relative permittivity for microstrip
    epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / w)**(-0.5)
    
    # Characteristic impedance (approximate)
    if w/h <= 1:
        Z0 = 60 / np.sqrt(epsilon_eff) * np.log(8 * h / w + w / (4 * h))
    else:
        Z0 = 120 * np.pi / np.sqrt(epsilon_eff) / (w / h + 1.393 + 0.667 * np.log(w / h + 1.444))
    
    # Capacitance per unit length
    C_prime = 1 / (Z0 * np.sqrt(epsilon_eff) * 3e8)  # c = 1/sqrt(LC), Z0 = sqrt(L/C)
    
    return C_prime

def calculate_capacitance_mom(Z, delta, V1=1.0):
    """
    Calculate capacitance per unit length using Method of Moments
    
    Parameters:
    Z (numpy.ndarray): impedance matrix
    delta (float): segment width
    V1 (float): applied voltage
    
    Returns:
    float: capacitance per unit length (F/m)
    """
    # Total charge on the strip
    N = Z.shape[0]
    V = np.ones(N) * V1  # Voltage vector
    I = solve_point_match_system(Z, V)  # Current distribution
    
    # Total charge per unit length
    Q_total = np.sum(I) * delta
    
    # Capacitance per unit length
    C_prime = Q_total / V1
    
    return C_prime

def analyze_surface_charge_distribution():
    """
    Find and plot surface charge distribution for different w/h ratios and N values
    """
    print("Surface Charge Distribution Analysis")
    print("=" * 50)
    
    # Parameters
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0]  # w/h < 1, = 1, > 1
    N_values = [5, 10, 20, 40]
    
    fig, axes = plt.subplots(len(w_h_ratios), len(N_values), figsize=(16, 12))
    if len(w_h_ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(N_values) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, w_h_ratio in enumerate(w_h_ratios):
        w = w_h_ratio * h
        
        for j, N in enumerate(N_values):
            # Calculate positions and impedance matrix
            x_positions = calculate_segment_positions(N, w)
            Z = populate_impedance_matrix(N, w, h, x_positions)
            
            # Calculate current and surface charge density
            V = np.ones(N)  # V1 = 1V
            I = solve_point_match_system(Z, V)
            delta = w / N
            rho_s = calculate_surface_charge_density(I, delta)
            
            # Plot
            axes[i, j].plot(x_positions, rho_s, 'bo-', linewidth=2, markersize=4)
            axes[i, j].set_title(f'w/h = {w_h_ratio}, N = {N}')
            axes[i, j].set_xlabel('Position (m)')
            axes[i, j].set_ylabel('Surface Charge Density (C/m²)')
            axes[i, j].grid(True, alpha=0.3)
    
    plt.suptitle('Surface Charge Distribution for Different w/h Ratios and N Values', fontsize=16)
    plt.tight_layout()
    plt.savefig('surface_charge_distribution.png', dpi=300, bbox_inches='tight')
    print("Surface charge distribution plot saved as 'surface_charge_distribution.png'")
    plt.close()

def convergence_analysis():
    """
    Analyze convergence of MoM results compared to empirical formulas
    """
    print("Convergence Analysis: MoM vs Empirical Formulas")
    print("=" * 60)
    
    # Parameters
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0, 5.0]
    N_values = [5, 10, 15, 20, 25, 30, 40, 50]
    
    # Storage for results
    results = []
    
    for w_h_ratio in w_h_ratios:
        w = w_h_ratio * h
        
        # Empirical capacitance
        C_empirical = empirical_capacitance_microstrip(w, h)
        
        print(f"\nw/h = {w_h_ratio}, w = {w*1000:.1f}mm, h = {h*1000:.1f}mm")
        print(f"Empirical C' = {C_empirical*1e12:.3f} pF/m")
        
        # MoM calculations for different N
        C_mom_values = []
        errors = []
        
        for N in N_values:
            try:
                # Calculate MoM capacitance
                x_positions = calculate_segment_positions(N, w)
                Z = populate_impedance_matrix(N, w, h, x_positions)
                delta = w / N
                C_mom = calculate_capacitance_mom(Z, delta)
                
                # Calculate relative error
                error = abs(C_mom - C_empirical) / C_empirical * 100
                
                C_mom_values.append(C_mom)
                errors.append(error)
                
                results.append({
                    'w/h': w_h_ratio,
                    'N': N,
                    'C_empirical (pF/m)': C_empirical * 1e12,
                    'C_MoM (pF/m)': C_mom * 1e12,
                    'Error (%)': error
                })
                
                print(f"  N = {N:2d}: C' = {C_mom*1e12:6.3f} pF/m, Error = {error:5.2f}%")
                
            except Exception as e:
                print(f"  N = {N:2d}: Failed - {str(e)}")
                C_mom_values.append(np.nan)
                errors.append(np.nan)
        
        # Plot convergence for this w/h ratio
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(N_values, [c*1e12 for c in C_mom_values], 'bo-', label='MoM')
        plt.axhline(y=C_empirical*1e12, color='r', linestyle='--', label='Empirical')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Capacitance (pF/m)')
        plt.title(f'Capacitance Convergence (w/h = {w_h_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(N_values, errors, 'ro-')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Relative Error (%)')
        plt.title(f'Convergence Error (w/h = {w_h_ratio})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'convergence_w_h_{w_h_ratio}.png', dpi=300, bbox_inches='tight')
        print(f"Convergence plot for w/h={w_h_ratio} saved as 'convergence_w_h_{w_h_ratio}.png'")
        plt.close()
    
    # Create summary table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SUMMARY TABLE: MoM vs Empirical Capacitance Results")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.3f'))
    
    return df

def create_non_uniform_segments(N, w, edge_density_factor=3.0):
    """
    Create non-uniform segmentation with higher density near edges
    
    Parameters:
    N (int): total number of segments
    w (float): width of microstrip
    edge_density_factor (float): factor to increase density near edges
    
    Returns:
    numpy.ndarray: segment positions
    numpy.ndarray: segment widths
    """
    # Create a non-uniform grid with higher density near edges
    # Use cosine spacing to concentrate points near edges
    theta = np.linspace(0, np.pi, N+1)
    x_uniform = np.cos(theta)  # Maps from [-1, 1]
    
    # Scale to actual width and sort
    x_positions = x_uniform * w/2
    x_positions = np.sort(x_positions)  # Ensure proper ordering
    
    # Calculate segment widths - each segment is between consecutive points
    segment_widths = np.diff(x_positions)
    
    # Use segment centers as the positions for MoM
    segment_centers = (x_positions[:-1] + x_positions[1:]) / 2
    
    return segment_centers, segment_widths

def analyze_high_resolution_current_distribution():
    """
    Analyze current distribution with high-resolution, non-uniform segmentation
    """
    print("High-Resolution Current Distribution Analysis")
    print("=" * 60)
    
    # Parameters for high-resolution analysis
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0, 5.0]
    N_values = [25, 40, 60, 80, 100]  # High-order N values
    
    fig, axes = plt.subplots(len(w_h_ratios), len(N_values), figsize=(20, 16))
    if len(w_h_ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(N_values) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, w_h_ratio in enumerate(w_h_ratios):
        w = w_h_ratio * h
        
        for j, N in enumerate(N_values):
            print(f"Processing w/h = {w_h_ratio}, N = {N}")
            
            # Create non-uniform segmentation
            x_positions, segment_widths = create_non_uniform_segments(N, w)
            
            # Create impedance matrix with non-uniform segments
            Z = np.zeros((N, N))
            epsilon_0 = 8.854187817e-12
            d = 2 * h
            
            # Populate impedance matrix for non-uniform segments
            for row in range(N):
                for col in range(N):
                    delta_i = segment_widths[row]  # Width of segment i
                    delta_j = segment_widths[col]  # Width of segment j
                    
                    # Use average segment width for K1 calculation
                    delta_avg = (delta_i + delta_j) / 2
                    
                    xi = x_positions[row]
                    xj = x_positions[col]
                    
                    term1 = K1(delta_avg/2, xi - xj, 0)
                    term2 = K1(delta_avg/2, xi - xj, d)
                    
                    Z[row, col] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
            
            # Calculate current distribution
            V = np.ones(N)  # V1 = 1V
            I = solve_point_match_system(Z, V)
            
            # Calculate surface charge density
            rho_s = I / segment_widths
            
            # Plot current distribution
            axes[i, j].plot(x_positions, I, 'bo-', linewidth=1.5, markersize=3)
            axes[i, j].set_title(f'w/h = {w_h_ratio}, N = {N}\nCurrent Distribution')
            axes[i, j].set_xlabel('Position (m)')
            axes[i, j].set_ylabel('Current (A)')
            axes[i, j].grid(True, alpha=0.3)
            
            # Add text box with key statistics
            max_current = np.max(I)
            min_current = np.min(I)
            current_range = max_current - min_current
            textstr = f'Max: {max_current:.3e}\nMin: {min_current:.3e}\nRange: {current_range:.3e}'
            axes[i, j].text(0.02, 0.98, textstr, transform=axes[i, j].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('High-Resolution Current Distribution Analysis\n(Non-Uniform Segmentation)', fontsize=16)
    plt.tight_layout()
    plt.savefig('high_resolution_current_distribution.png', dpi=300, bbox_inches='tight')
    print("High-resolution current distribution plot saved as 'high_resolution_current_distribution.png'")
    plt.close()

def analyze_edge_effects():
    """
    Analyze edge effects and charge concentration with high N values
    """
    print("\nEdge Effects Analysis")
    print("=" * 40)
    
    # Parameters
    h = 0.001
    w_h_ratios = [1.0, 2.0, 5.0]
    N_values = [50, 80, 120]
    
    fig, axes = plt.subplots(len(w_h_ratios), len(N_values), figsize=(18, 12))
    if len(w_h_ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(N_values) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, w_h_ratio in enumerate(w_h_ratios):
        w = w_h_ratio * h
        
        for j, N in enumerate(N_values):
            print(f"Analyzing edge effects: w/h = {w_h_ratio}, N = {N}")
            
            # Create non-uniform segmentation
            x_positions, segment_widths = create_non_uniform_segments(N, w, edge_density_factor=4.0)
            
            # Create impedance matrix
            Z = np.zeros((N, N))
            epsilon_0 = 8.854187817e-12
            d = 2 * h
            
            for row in range(N):
                for col in range(N):
                    delta_avg = (segment_widths[row] + segment_widths[col]) / 2
                    xi = x_positions[row]
                    xj = x_positions[col]
                    
                    term1 = K1(delta_avg/2, xi - xj, 0)
                    term2 = K1(delta_avg/2, xi - xj, d)
                    
                    Z[row, col] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
            
            # Calculate current and charge density
            V = np.ones(N)
            I = solve_point_match_system(Z, V)
            rho_s = I / segment_widths
            
            # Plot surface charge density (normalized)
            rho_s_normalized = rho_s / np.max(rho_s)
            axes[i, j].plot(x_positions, rho_s_normalized, 'ro-', linewidth=2, markersize=4)
            axes[i, j].set_title(f'w/h = {w_h_ratio}, N = {N}\nNormalized Charge Density')
            axes[i, j].set_xlabel('Position (m)')
            axes[i, j].set_ylabel('Normalized ρs')
            axes[i, j].grid(True, alpha=0.3)
            
            # Highlight edge regions
            edge_threshold = 0.1 * w  # 10% of width from edges
            axes[i, j].axvspan(-w/2, -w/2 + edge_threshold, alpha=0.2, color='red', label='Left Edge')
            axes[i, j].axvspan(w/2 - edge_threshold, w/2, alpha=0.2, color='red', label='Right Edge')
            
            # Calculate edge enhancement factor
            center_indices = np.abs(x_positions) < w/4  # Center 50% of strip
            edge_indices = (np.abs(x_positions) > w/2 - edge_threshold)
            
            if np.any(center_indices) and np.any(edge_indices):
                center_charge = np.mean(rho_s[center_indices])
                edge_charge = np.mean(rho_s[edge_indices])
                enhancement_factor = edge_charge / center_charge if center_charge > 0 else 1.0
                
                textstr = f'Edge Enhancement:\n{enhancement_factor:.2f}x'
                axes[i, j].text(0.02, 0.98, textstr, transform=axes[i, j].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Edge Effects Analysis - Charge Concentration at Strip Edges', fontsize=16)
    plt.tight_layout()
    plt.savefig('edge_effects_analysis.png', dpi=300, bbox_inches='tight')
    print("Edge effects analysis plot saved as 'edge_effects_analysis.png'")
    plt.close()

def convergence_high_n():
    """
    Convergence analysis focusing on high N values
    """
    print("\nHigh-N Convergence Analysis")
    print("=" * 40)
    
    h = 0.001
    w_h_ratios = [1.0, 2.0, 5.0]
    N_values = [20, 30, 40, 50, 60, 80, 100, 120]
    
    results = []
    
    for w_h_ratio in w_h_ratios:
        w = w_h_ratio * h
        
        # Empirical capacitance
        C_empirical = empirical_capacitance_microstrip(w, h)
        
        print(f"\nw/h = {w_h_ratio}, Empirical C' = {C_empirical*1e12:.3f} pF/m")
        
        C_mom_values = []
        errors = []
        
        for N in N_values:
            try:
                # Use non-uniform segmentation for high N
                if N >= 50:
                    x_positions, segment_widths = create_non_uniform_segments(N, w)
                    Z = np.zeros((N, N))
                    epsilon_0 = 8.854187817e-12
                    d = 2 * h
                    
                    for row in range(N):
                        for col in range(N):
                            delta_avg = (segment_widths[row] + segment_widths[col]) / 2
                            xi = x_positions[row]
                            xj = x_positions[col]
                            
                            term1 = K1(delta_avg/2, xi - xj, 0)
                            term2 = K1(delta_avg/2, xi - xj, d)
                            
                            Z[row, col] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
                    
                    # Calculate capacitance
                    V = np.ones(N)
                    I = solve_point_match_system(Z, V)
                    Q_total = np.sum(I * segment_widths)
                    C_mom = Q_total
                    
                else:
                    # Use uniform segmentation for lower N
                    x_positions = calculate_segment_positions(N, w)
                    Z = populate_impedance_matrix(N, w, h, x_positions)
                    delta = w / N
                    C_mom = calculate_capacitance_mom(Z, delta)
                
                error = abs(C_mom - C_empirical) / C_empirical * 100
                
                C_mom_values.append(C_mom)
                errors.append(error)
                
                results.append({
                    'w/h': w_h_ratio,
                    'N': N,
                    'C_empirical (pF/m)': C_empirical * 1e12,
                    'C_MoM (pF/m)': C_mom * 1e12,
                    'Error (%)': error
                })
                
                print(f"  N = {N:3d}: C' = {C_mom*1e12:7.3f} pF/m, Error = {error:5.2f}%")
                
            except Exception as e:
                print(f"  N = {N:3d}: Failed - {str(e)}")
                C_mom_values.append(np.nan)
                errors.append(np.nan)
        
        # Plot convergence
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(N_values, [c*1e12 for c in C_mom_values], 'bo-', label='MoM (Non-uniform for N≥50)')
        plt.axhline(y=C_empirical*1e12, color='r', linestyle='--', label='Empirical')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Capacitance (pF/m)')
        plt.title(f'High-N Convergence (w/h = {w_h_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(N_values, errors, 'ro-')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Relative Error (%)')
        plt.title(f'High-N Error Convergence (w/h = {w_h_ratio})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'high_n_convergence_w_h_{w_h_ratio}.png', dpi=300, bbox_inches='tight')
        print(f"High-N convergence plot for w/h={w_h_ratio} saved")
        plt.close()
    
    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("HIGH-N CONVERGENCE SUMMARY")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.3f'))
    
    return df

def main():
    """
    Main function to run the complete microstrip analysis
    """
    print("Microstrip Line Analysis using Method of Moments")
    print("=" * 60)
    print("Analysis includes:")
    print("1. Surface charge distribution for different w/h ratios and N values")
    print("2. Capacitance convergence analysis")
    print("3. Comparison with empirical formulas")
    print("4. High-resolution current distribution analysis")
    print("5. Edge effects analysis")
    print("6. High-N convergence analysis")
    print()
    
    # Run original analyses
    analyze_surface_charge_distribution()
    results_df = convergence_analysis()
    
    # Run high-resolution analyses
    analyze_high_resolution_current_distribution()
    analyze_edge_effects()
    high_n_results = convergence_high_n()
    
    return results_df, high_n_results

def test_K1_function():
    """
    Test function to verify K1 implementation
    """
    print("Testing K1 function...")
    
    # Test cases
    test_cases = [
        (0.1, 0.0, 0.0),    # a=0.1, x=0, y=0
        (0.1, 0.05, 0.0),   # a=0.1, x=0.05, y=0
        (0.1, 0.0, 0.1),    # a=0.1, x=0, y=0.1
    ]
    
    for a, x, y in test_cases:
        result = K1(a, x, y)
        print(f"K1(a={a}, x={x}, y={y}) = {result:.6f}")
    
    print()

if __name__ == "__main__":
    # Run tests
    test_K1_function()
    
    # Run main analysis
    results_df = main()
