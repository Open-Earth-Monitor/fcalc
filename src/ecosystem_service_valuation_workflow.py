#!/usr/bin/env python3
"""
Ecosystem Service Valuation Workflow Implementation

This module implements a generic valuation workflow for ecosystem services.
It calculates the economic benefits of interventions by analyzing changes in 
model outputs and translating these into monetary values, supporting 
land-use specific conversion factors.

Author: Manus AI
Date: June 19, 2025
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class EcosystemServiceValuationWorkflow:
    """
    A class to implement a generic ecosystem service valuation workflow.
    It can handle various model outputs and supports land-use specific unit-to-monetary conversions.
    """
    
    def __init__(self, discount_rate: float = 0.03):
        """
        Initialize the Ecosystem Service Valuation Workflow.
        
        Args:
            discount_rate (float): Discount rate for NPV calculations. Default is 3% (0.03).
        """
        self.discount_rate = discount_rate
        self.baseline_data = None
        self.baseline_metadata = None
        self.intervention_data = {}
        self.delta_service_provision = {}
        self.monetary_value_maps = {}
        self.financial_metrics = {}
        self.land_use_map = None
        self.land_use_metadata = None
        self.service_unit = "units" # Initialize service_unit here
        
    def load_raster(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a raster file and return the data array and metadata.
        
        Args:
            file_path (str): Path to the raster file.
            
        Returns:
            Tuple[np.ndarray, dict]: Data array and metadata dictionary.
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read the first band
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'nodata': src.nodata
                }
                return data, metadata
        except Exception as e:
            raise FileNotFoundError(f"Error loading raster file {file_path}: {str(e)}")
    
    def load_baseline_scenario(self, baseline_path: str, service_unit: str = "units"):
        """
        Load the baseline scenario model output raster.
        
        Args:
            baseline_path (str): Path to the baseline model output file.
            service_unit (str): Unit of the ecosystem service (e.g., "m³", "tons", "units").
        """
        print(f"Loading baseline scenario from: {baseline_path}")
        self.baseline_data, self.baseline_metadata = self.load_raster(baseline_path)
        self.service_unit = service_unit
        print(f"Baseline data shape: {self.baseline_data.shape}")
        print(f"Baseline data range: {np.nanmin(self.baseline_data):.2f} to {np.nanmax(self.baseline_data):.2f} {self.service_unit}")
        
    def load_intervention_scenario(self, intervention_path: str, scenario_name: str):
        """
        Load an intervention scenario model output raster.
        
        Args:
            intervention_path (str): Path to the intervention model output file.
            scenario_name (str): Name identifier for the intervention scenario.
        """
        print(f"Loading intervention scenario '{scenario_name}' from: {intervention_path}")
        data, metadata = self.load_raster(intervention_path)
        self.intervention_data[scenario_name] = data
        print(f"Intervention '{scenario_name}' data shape: {data.shape}")
        print(f"Intervention '{scenario_name}' data range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f} {self.service_unit}")
        
    def load_land_use_map(self, land_use_path: str):
        """
        Load a land use map raster.
        
        Args:
            land_use_path (str): Path to the land use map file.
        """
        print(f"Loading land use map from: {land_use_path}")
        self.land_use_map, self.land_use_metadata = self.load_raster(land_use_path)
        print(f"Land use map shape: {self.land_use_map.shape}")
        print(f"Land use map unique values: {np.unique(self.land_use_map)}")

    def calculate_delta_service_provision(self, scenario_name: str):
        """
        Calculate the change in ecosystem service provision for a specific intervention scenario.
        
        Args:
            scenario_name (str): Name of the intervention scenario.
        """
        if self.baseline_data is None:
            raise ValueError("Baseline scenario must be loaded first.")
        
        if scenario_name not in self.intervention_data:
            raise ValueError(f"Intervention scenario '{scenario_name}' not found.")
        
        # Calculate delta service provision: intervention - baseline
        delta_s = self.intervention_data[scenario_name] - self.baseline_data
        self.delta_service_provision[scenario_name] = delta_s
        
        # Calculate statistics
        total_delta = np.nansum(delta_s)
        positive_delta = np.nansum(delta_s[delta_s > 0])
        negative_delta = np.nansum(delta_s[delta_s < 0])
        
        print(f"Delta service provision for '{scenario_name}':")
        print(f"  Total change: {total_delta:.2f} {self.service_unit}")
        print(f"  Positive change (increased provision): {positive_delta:.2f} {self.service_unit}")
        print(f"  Negative change (decreased provision): {negative_delta:.2f} {self.service_unit}")
        
    def calculate_monetary_value_map(self, scenario_name: str, 
                                     conversion_factors: Union[float, Dict[int, float]]):
        """
        Calculate the monetary value map for a specific intervention scenario.
        
        Args:
            scenario_name (str): Name of the intervention scenario.
            conversion_factors (Union[float, Dict[int, float]]): 
                A single float for a global conversion factor (e.g., €/unit) or 
                a dictionary mapping land use codes (int) to conversion factors (float).
        """
        if scenario_name not in self.delta_service_provision:
            raise ValueError(f"Delta service provision for scenario '{scenario_name}' not calculated yet.")
        
        delta_s = self.delta_service_provision[scenario_name]
        monetary_value = np.zeros_like(delta_s, dtype=float)

        if isinstance(conversion_factors, (int, float)):
            # Global conversion factor
            monetary_value = delta_s * conversion_factors
            print(f"Applying global conversion factor: {conversion_factors} €/{self.service_unit}")
        elif isinstance(conversion_factors, dict):
            # Land-use specific conversion factors
            if self.land_use_map is None:
                raise ValueError("Land use map must be loaded for land-use specific conversion factors.")
            
            print("Applying land-use specific conversion factors.")
            for lu_code, factor in conversion_factors.items():
                # Apply factor only where land use matches and delta_s is not NaN
                mask = (self.land_use_map == lu_code) & (~np.isnan(delta_s))
                monetary_value[mask] = delta_s[mask] * factor
                print(f"  Land Use {lu_code}: {factor} €/{self.service_unit}")
        else:
            raise TypeError("conversion_factors must be a float or a dictionary.")

        self.monetary_value_maps[scenario_name] = monetary_value
        
        total_monetary_value = np.nansum(monetary_value)
        print(f"Monetary value map for '{scenario_name}':")
        print(f"  Total monetary value: €{total_monetary_value:.2f}")
        
    def calculate_financial_metrics(self, scenario_name: str, investment_cost: float, 
                                  project_lifespan: int = 20, annual_maintenance_cost: float = 0.0):
        """
        Calculate financial metrics for a specific intervention scenario.
        
        Args:
            scenario_name (str): Name of the intervention scenario.
            investment_cost (float): Total initial investment cost in €.
            project_lifespan (int): Project lifespan in years. Default is 20 years.
            annual_maintenance_cost (float): Annual maintenance cost in €. Default is 0.
        """
        if scenario_name not in self.monetary_value_maps:
            raise ValueError(f"Monetary value map for scenario '{scenario_name}' not calculated yet.")
        
        total_monetary_value = np.nansum(self.monetary_value_maps[scenario_name])
        
        # Calculate BCR (Benefit-Cost Ratio)
        bcr = total_monetary_value / investment_cost if investment_cost > 0 else float('inf')
        
        # Calculate NPV (Net Present Value)
        annual_benefit = total_monetary_value  # Assuming benefits occur annually
        npv = 0
        for year in range(project_lifespan):
            annual_net_benefit = annual_benefit - annual_maintenance_cost
            if year == 0:
                annual_net_benefit -= investment_cost  # Initial investment in year 0
            npv += annual_net_benefit / ((1 + self.discount_rate) ** year)
        
        # Calculate Payback Period
        payback_period = investment_cost / annual_benefit if annual_benefit > 0 else float('inf')
        
        # Store metrics
        self.financial_metrics[scenario_name] = {
            'total_monetary_value': total_monetary_value,
            'investment_cost': investment_cost,
            'annual_maintenance_cost': annual_maintenance_cost,
            'bcr': bcr,
            'npv': npv,
            'payback_period': payback_period,
            'project_lifespan': project_lifespan
        }
        
        print(f"Financial metrics for '{scenario_name}':")
        print(f"  Total Monetary Value: €{total_monetary_value:.2f}")
        print(f"  Investment Cost: €{investment_cost:.2f}")
        print(f"  Benefit-Cost Ratio (BCR): {bcr:.2f}")
        print(f"  Net Present Value (NPV): €{npv:.2f}")
        print(f"  Payback Period: {payback_period:.1f} years")
        
    def visualize_monetary_value_map(self, scenario_name: str, save_path: Optional[str] = None):
        """
        Visualize the monetary value map for a specific intervention scenario.
        
        Args:
            scenario_name (str): Name of the intervention scenario.
            save_path (Optional[str]): Path to save the visualization. If None, displays the plot.
        """
        if scenario_name not in self.monetary_value_maps:
            raise ValueError(f"Monetary value map for scenario '{scenario_name}' not calculated yet.")
        
        monetary_value = self.monetary_value_maps[scenario_name]
        
        plt.figure(figsize=(12, 8))
        
        # Create the heatmap
        im = plt.imshow(monetary_value, cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(im, label='Monetary Value (€)')
        plt.title(f'Monetary Value Map - {scenario_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Pixel X', fontsize=12)
        plt.ylabel('Pixel Y', fontsize=12)
        
        # Add statistics text
        total_value = np.nansum(monetary_value)
        max_value = np.nanmax(monetary_value)
        min_value = np.nanmin(monetary_value)
        
        stats_text = f'Total: €{total_value:.0f}\nMax: €{max_value:.2f}\nMin: €{min_value:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Monetary value map saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def compare_scenarios(self, save_path: Optional[str] = None):
        """
        Create a comparison chart of financial metrics across all scenarios.
        
        Args:
            save_path (Optional[str]): Path to save the comparison chart. If None, displays the plot.
        """
        if not self.financial_metrics:
            raise ValueError("No financial metrics calculated yet.")
        
        # Prepare data for plotting
        scenarios = list(self.financial_metrics.keys())
        bcr_values = [self.financial_metrics[s]['bcr'] for s in scenarios]
        npv_values = [self.financial_metrics[s]['npv'] for s in scenarios]
        payback_values = [self.financial_metrics[s]['payback_period'] for s in scenarios]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # BCR comparison
        bars1 = axes[0].bar(scenarios, bcr_values, color='skyblue', alpha=0.7)
        axes[0].set_title('Benefit-Cost Ratio (BCR)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('BCR')
        axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even (BCR=1)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, bcr_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # NPV comparison
        bars2 = axes[1].bar(scenarios, npv_values, color='lightgreen', alpha=0.7)
        axes[1].set_title('Net Present Value (NPV)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('NPV (€)')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even (NPV=0)')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, npv_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(npv_values)*0.01,
                        f'€{value:.0f}', ha='center', va='bottom')
        
        # Payback Period comparison
        bars3 = axes[2].bar(scenarios, payback_values, color='orange', alpha=0.7)
        axes[2].set_title('Payback Period', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Years')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars3, payback_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}y', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scenario comparison chart saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate a summary report of all calculated financial metrics.
        
        Returns:
            pd.DataFrame: Summary report as a pandas DataFrame.
        """
        if not self.financial_metrics:
            raise ValueError("No financial metrics calculated yet.")
        
        # Create summary DataFrame
        summary_data = []
        for scenario, metrics in self.financial_metrics.items():
            summary_data.append({
                'Scenario': scenario,
                'Total Monetary Value (€)': f"{metrics['total_monetary_value']:.2f}",
                'Investment Cost (€)': f"{metrics['investment_cost']:.2f}",
                'BCR': f"{metrics['bcr']:.2f}",
                'NPV (€)': f"{metrics['npv']:.2f}",
                'Payback Period (years)': f"{metrics['payback_period']:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
        
    def save_results(self, output_dir: str):
        """
        Save all results including monetary value maps, comparison charts, and summary report.
        
        Args:
            output_dir (str): Directory to save all output files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save monetary value maps for each scenario
        for scenario in self.monetary_value_maps.keys():
            map_path = output_path / f"monetary_value_map_{scenario}.png"
            self.visualize_monetary_value_map(scenario, str(map_path))
        
        # Save scenario comparison chart
        comparison_path = output_path / "scenario_comparison.png"
        self.compare_scenarios(str(comparison_path))
        
        # Save summary report
        summary_df = self.generate_summary_report()
        summary_path = output_path / "financial_summary_report.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Financial summary report saved to: {summary_path}")
        
        print(f"All results saved to: {output_path}")


def create_sample_data():
    """
    Create sample raster data for demonstration purposes.
    This function generates synthetic data that mimics generic model outputs.
    """
    print("Creating sample data for demonstration...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define raster dimensions
    height, width = 100, 100
    
    # Create baseline scenario (e.g., current ecosystem service provision)
    baseline = np.random.uniform(10, 50, (height, width))  # Example: 10-50 units of service
    
    # Create intervention scenarios
    # Scenario 1: Moderate improvement
    intervention1 = baseline + np.random.uniform(5, 15, (height, width))
    
    # Scenario 2: Significant improvement
    intervention2 = baseline + np.random.uniform(10, 25, (height, width))
    
    # Scenario 3: Targeted improvement
    intervention3 = baseline + np.random.uniform(3, 10, (height, width))
    
    # Create a sample land use map
    land_use_map = np.random.randint(1, 5, (height, width)) # 4 land use types (1, 2, 3, 4)
    
    # Save as GeoTIFF files (simplified - in practice, would include proper georeferencing)
    scenarios = {
        'baseline': baseline,
        'intervention_A': intervention1,
        'intervention_B': intervention2,
        'intervention_C': intervention3
    }
    
    # Create a simple transform and CRS for the sample data
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    
    transform = from_bounds(0, 0, width, height, width, height)
    crs = CRS.from_epsg(4326)  # WGS84
    
    for name, data in scenarios.items():
        filename = f"sample_{name}_service_provision.tif"
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        print(f"Created sample file: {filename}")
        
    # Save land use map
    land_use_filename = "sample_land_use_map.tif"
    with rasterio.open(
        land_use_filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=land_use_map.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(land_use_map, 1)
    print(f"Created sample file: {land_use_filename}")
    
    return list(scenarios.keys()), land_use_filename


if __name__ == "__main__":
    # Demonstration of the Generic Ecosystem Service Valuation Workflow
    print("=== Generic Ecosystem Service Valuation Workflow Demonstration ===\n")
    
    # Create sample data
    scenario_names, land_use_map_path = create_sample_data()
    
    # Initialize the workflow
    workflow = EcosystemServiceValuationWorkflow(discount_rate=0.03)
    
    # Load baseline scenario
    workflow.load_baseline_scenario("sample_baseline_service_provision.tif", service_unit="units")
    
    # Load intervention scenarios
    intervention_scenarios = [s for s in scenario_names if s != 'baseline']
    for scenario in intervention_scenarios:
        workflow.load_intervention_scenario(f"sample_{scenario}_service_provision.tif", scenario)
    
    # Load land use map for land-use specific conversions
    workflow.load_land_use_map(land_use_map_path)

    # Define investment costs for each scenario (example values in €)
    investment_costs = {
        'intervention_A': 75000,
        'intervention_B': 120000,
        'intervention_C': 60000
    }
    
    # Define conversion factors
    # Example 1: Global conversion factor
    global_conversion_factor = 5.0  # € per unit of service
    
    # Example 2: Land-use specific conversion factors
    # Assuming land use codes 1, 2, 3, 4
    land_use_conversion_factors = {
        1: 3.0,  # €/unit for land use type 1
        2: 7.0,  # €/unit for land use type 2
        3: 4.5,  # €/unit for land use type 3
        4: 6.0   # €/unit for land use type 4
    }

    # Process each intervention scenario
    for scenario in intervention_scenarios:
        print(f"\n--- Processing {scenario} ---")
        
        # Calculate delta service provision
        workflow.calculate_delta_service_provision(scenario)
        
        # Calculate monetary value map using land-use specific factors
        workflow.calculate_monetary_value_map(scenario, land_use_conversion_factors)
        # To use global conversion factor, uncomment the line below and comment the above line:
        # workflow.calculate_monetary_value_map(scenario, global_conversion_factor)
        
        # Calculate financial metrics
        workflow.calculate_financial_metrics(
            scenario, 
            investment_costs[scenario], 
            project_lifespan=20,
            annual_maintenance_cost=investment_costs[scenario] * 0.015  # 1.5% annual maintenance
        )
    
    # Generate visualizations and reports
    print("\n--- Generating Results ---")
    workflow.save_results("ecosystem_service_results")
    
    # Display summary report
    print("\n--- Financial Summary Report ---")
    summary_df = workflow.generate_summary_report()
    print(summary_df.to_string(index=False))
    
    print("\n=== Workflow Complete ===")


