# Financial Calculator for Ecosystem Services: Documentation

**A Flexible Solution for Valuing Diverse Ecosystem Services**

---

**Author:** Rolf Simoes  
**Date:** June 19, 2025  
**Version:** 1.0

---

## Executive Summary

This document introduces `fcalc`, a tool for estimating the economic value of ecosystem service interventions. It is based on established methods for ecosystem service valuation and supports the conversion of model outputs into monetary terms, including the use of land-use-specific cost factors.

`fcalc` is designed to be flexible. It can handle outputs from various models and apply a standardized valuation logic across multiple ecosystem services, such as flood mitigation, soil erosion control, water purification, and carbon storage. The tool is implemented in Python and supports geospatial processing, numerical analysis, and basic financial evaluation.

The implementation includes a modular structure that calculates service changes and converts them into economic values. It supports both fixed and land-use-dependent conversion rates. The tool provides common financial indicators such as Benefit-Cost Ratio (BCR), Net Present Value (NPV), and Payback Period. It has been tested in different scenarios and shown to be adaptable and reliable.

---

## Table of Contents

1. [Introduction and Background](#introduction-and-background)
2. [Methodology and Theoretical Framework](#methodology-and-theoretical-framework)
3. [Implementation Architecture](#implementation-architecture)
4. [Technical Implementation](#technical-implementation)
5. [Testing and Validation](#testing-and-validation)
6. [Results and Analysis](#results-and-analysis)
7. [Usage Guidelines](#usage-guidelines)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)
9. [References](#references)

---

## Introduction and Background

Ecosystems support human well-being by providing services such as air and water regulation, pollination, and protection from natural hazards. These services often go unaccounted for in economic planning, despite their importance. Quantifying their value is essential for supporting conservation and guiding sustainable development.

**Ecosystem service valuation** assigns monetary value to the benefits ecosystems provide. This allows decision-makers to compare the costs and benefits of conservation, restoration, and land-use change. While the concept is widely recognized, practical tools are often limited by the diversity of services, the complexity of ecological models, and the need for locally relevant data.

This document presents a financial calculator designed to estimate the economic value of changes in ecosystem service provision. It supports outputs from multiple models and uses both fixed and land-use-specific conversion factors. The tool aims to provide a consistent and transparent method for valuation across services such as flood mitigation, erosion control, and water quality improvement.

This document presents the design, implementation, and validation of the `fcalc` ecosystem service valuation tool. It describes the theoretical framework, system architecture, and technical components used to support spatial valuation across multiple services. The tool is intended to help integrate ecosystem service assessments into environmental planning and policy by enabling consistent and transparent economic analysis.


Here is an improved version of your **“Methodology and Theoretical Framework”** section. It preserves all essential concepts, removes repetition, simplifies sentence structure, and maintains a clear, professional tone:

---

## Methodology and Theoretical Framework

### Ecosystem Service Valuation Principles

Valuing ecosystem services involves estimating the economic contribution of natural systems to human well-being. By expressing these contributions in monetary terms, the approach allows comparison with the costs of interventions or alternative land-use strategies. The guiding assumption is that if a natural process provides a benefit that would otherwise require human-built solutions or lead to damage, it has economic value.

Valuation methods include revealed preference (e.g., hedonic pricing, travel cost), stated preference (e.g., contingent valuation, choice experiments), and production function approaches. This work applies a **production function approach**, specifically the **avoided cost method**, which estimates the value of services that reduce or prevent economic losses.

### Difference-Based Valuation Approach

The workflow adopts a **difference-based valuation approach**, which compares ecosystem service provision between a baseline scenario and one or more intervention scenarios. The difference in service levels is interpreted as a gain or loss with potential economic implications.

Let $S_{\text{intervention}}(x,y)$ and $S_{\text{baseline}}(x,y)$ represent the level of service in the intervention and baseline scenarios, respectively, at location $(x, y)$. The change in service provision is:

$$
\Delta S(x,y) = S_{\text{intervention}}(x,y) - S_{\text{baseline}}(x,y)
$$

This change can be positive (benefit) or negative. To estimate the monetary value, a **unit conversion factor** is applied, representing the economic value per unit of service:

$$
\text{Value}(x,y) = \Delta S(x,y) \times \text{Conversion Factor}
$$

The framework is adaptable to different services. For example:

* If the service is **avoided soil erosion**, $\Delta S$ is in tons of soil, and the conversion factor is €/ton.
* For **water purification**, $\Delta S$ may be in cubic meters of treated water, with €/m³ as the unit value.
* For **flood mitigation**, $\Delta S$ is in cubic meters of runoff retained, valued in €/m³ of avoided flood damage.

This structure enables the valuation of spatially distributed ecosystem services using consistent, transparent assumptions. The results can inform investment prioritization, cost-benefit analysis, and long-term planning.

### Land-Use Specific Conversion Factors

This framework supports the use of **land-use specific conversion factors**, recognizing that the economic value of a service can vary depending on where it is delivered. For instance, retaining one cubic meter of stormwater in a dense urban area may prevent more damage—and thus have a higher economic value—than in a rural setting.

To enable this, the valuation process incorporates a land-use raster aligned with the spatial resolution of the ecosystem service data. A conversion table or dictionary maps each land-use code to a monetary value per unit of service. The per-pixel valuation is then calculated as:

$$
\text{Value}(x, y) = \Delta S(x, y) \times \text{Conversion Factor}_{LU(x, y)}
$$

Where $LU(x, y)$ is the land-use class at pixel $(x, y)$, and the corresponding conversion factor reflects its economic context.

This approach increases the spatial precision of the valuation and allows for more realistic estimates in heterogeneous landscapes. It is particularly useful in applications where the value of avoided damages or benefits is sensitive to local land use, such as flood risk reduction, infrastructure protection, or urban ecosystem restoration.

### Financial Analysis Metrics

To assess the economic performance of ecosystem service interventions, the valuation process computes financial metrics using **per-pixel estimates** of avoided costs. These metrics are then aggregated to support decision-making at project or regional scales.

* **Benefit-Cost Ratio (BCR):**
  For each pixel, the avoided cost is compared to the portion of the intervention cost allocated to that location. The total BCR is computed by summing values over the analysis area:

  $$
  \text{BCR} = \frac{\sum_{x,y} \text{Monetary Value}(x, y)}{\sum_{x,y} \text{Investment Cost}(x, y)}
  $$

  A BCR greater than 1 indicates that the intervention yields more benefits than costs.

* **Net Present Value (NPV):**
  NPV discounts future benefits and costs at each pixel to account for the time value of money. The aggregated NPV across space and time is calculated as:

  $$
  \text{NPV} = \sum_{x,y} \sum_{t=0}^{T} \frac{B_t(x, y) - C_t(x, y)}{(1 + r)^t}
  $$

  Where:

  * $B_t(x, y)$: benefit at pixel $(x, y)$ in year $t$
  * $C_t(x, y)$: cost at pixel $(x, y)$ in year $t$
  * $r$: discount rate
  * $T$: project lifetime in years

* **Payback Period:**
  This metric estimates how quickly benefits accumulate to offset initial investment. Calculated per pixel and summarized across the area:

  $$
  \text{Payback Period}(x, y) = \frac{\text{Initial Cost}(x, y)}{\text{Annual Benefit}(x, y)}
  $$

  Aggregate statistics (e.g., mean or percentile payback time) can be used to compare scenarios.


You're right — the tool name `fcalc` should be explicitly referenced throughout the section to anchor the description and maintain consistency. Here's the improved version with **`fcalc` clearly named**, while keeping the style concise, neutral, and technically structured:

---

## Implementation Architecture

### System Design Overview

`fcalc` is implemented as a modular, object-oriented Python tool for spatial valuation of ecosystem services. It is organized around a central class, `EcosystemServiceValuationWorkflow`, and structured as a multi-stage pipeline. The system processes geospatial model outputs, applies valuation logic, and generates financial assessments to support scenario comparison and investment analysis.

### Core Components

**Data Management Module**
This module handles the input of raster datasets, including baseline and intervention outputs from ecosystem service models, as well as optional land-use rasters. It uses the `rasterio` library to ensure correct georeferencing, alignment, and metadata handling. Basic error checks are included for file integrity and spatial consistency.

**Service Change Calculation Engine**
This component computes the per-pixel difference in service provision between scenarios:

$$
\Delta S(x, y) = S_{\text{intervention}}(x, y) - S_{\text{baseline}}(x, y)
$$

The engine supports any quantitative, spatially distributed ecosystem service, such as runoff retention or sediment export.

**Monetary Valuation Engine**
This module converts service changes into monetary values. It supports:

* **Global conversion factors**, applying a uniform €/unit rate across all pixels.
* **Land-use-specific factors**, where each land-use type is assigned a different value.

This enables context-sensitive valuation based on the spatial distribution of land cover or economic exposure.

**Financial Analysis Module**
`fcalc` aggregates the pixel-wise monetary values and computes standard financial metrics:

* Benefit-Cost Ratio (BCR)
* Net Present Value (NPV)
* Payback Period

These are calculated based on investment costs, maintenance expenses, and discount rates, allowing users to evaluate the economic feasibility of interventions.

**Visualization and Reporting Tools**
This component generates spatial outputs such as heatmaps of monetary value and comparative plots across scenarios. Tabular summaries are also produced to support reporting and integration into broader planning processes.


### Data Flow and Interactions

The `fcalc` workflow begins by loading a baseline raster. This raster represents the reference level of ecosystem service provision. One or more intervention scenario rasters are then loaded. If land-use-specific valuation is used, a land-use raster is also included. All rasters are checked for consistent extent, resolution, and coordinate system.

The service change is calculated by subtracting the baseline from each intervention scenario. This result is passed to the valuation module. There, it is converted into a monetary value using either a single conversion factor or one based on land use.

The tool then aggregates the monetary values across all pixels. These totals, together with investment and maintenance costs, are used to compute financial metrics: Benefit-Cost Ratio, Net Present Value, and Payback Period.

The final step produces maps, charts, and summary tables. These outputs help compare the economic effects of different scenarios.


### Flexibility and Application Scope

The flexibility of `fcalc` comes from its abstraction of both the service unit and the conversion method. The service unit can be any measurable quantity, such as cubic meters, tons, kilograms, or unitless model scores. The method for calculating change uses a simple difference, making it applicable to any model that outputs comparable values under different scenarios.

The monetary valuation function accepts either a single value or a dictionary of land-use-specific rates. This allows the tool to adapt to different contexts without changing the core logic.

Example applications include:

* **Flood Risk Mitigation:** using m³ of retained runoff and €/m³ avoided damage.
* **Soil Erosion Control:** using tons of retained soil and €/ton avoided loss.
* **Water Quality Improvement:** using kg of pollutant removed and €/kg saved in treatment costs.

By keeping service-specific parameters separate from the core functions, `fcalc` can be applied across various ecosystem services with minimal adjustments.

---

## Technical Implementation

The `fcalc` tool is implemented in Python. It uses standard libraries for geospatial data processing (`rasterio`, `numpy`), numerical analysis, and visualization. The core logic is organized within a single class, `EcosystemServiceValuationWorkflow`, following an object-oriented structure.

### Core Class: `EcosystemServiceValuationWorkflow`

This class manages the full valuation process. It initializes key parameters and stores the data needed throughout the workflow. The constructor sets the discount rate and defines internal variables to hold rasters, service changes, monetary values, and financial results. It also includes a `service_unit` attribute that specifies the unit used for the ecosystem service (e.g., m³, tons, kg).

```python
class EcosystemServiceValuationWorkflow:
    def __init__(self, discount_rate: float = 0.03):
        self.discount_rate = discount_rate
        self.baseline_data = None
        self.baseline_metadata = None
        self.intervention_data = {}
        self.delta_service_provision = {}
        self.monetary_value_maps = {}
        self.financial_metrics = {}
        self.land_use_map = None
        self.land_use_metadata = None
        self.service_unit = "units"  # Default unit; can be updated later
```

Each method in the class performs a specific task: loading input data, calculating service changes, applying valuation, or computing financial metrics. This modular structure supports clarity, reuse, and adaptation to different services and contexts.


### Raster Data Handling

Raster data is central to the `fcalc` workflow. The `rasterio` library is used to read and write GeoTIFF files. It provides access to pixel values and metadata such as coordinate reference system, resolution, and dimensions.

The `load_raster` method reads a single-band raster and returns both the data array and a metadata dictionary:

```python
def load_raster(self, file_path: str) -> Tuple[np.ndarray, dict]:
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
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
```

The `load_baseline_scenario` method loads the baseline raster and sets the unit used throughout the valuation:

```python
def load_baseline_scenario(self, baseline_path: str, service_unit: str = "units"):
    print(f"Loading baseline scenario from: {baseline_path}")
    self.baseline_data, self.baseline_metadata = self.load_raster(baseline_path)
    self.service_unit = service_unit
    print(f"Baseline data shape: {self.baseline_data.shape}")
    print(f"Baseline data range: {np.nanmin(self.baseline_data):.2f} to {np.nanmax(self.baseline_data):.2f} {self.service_unit}")
```

The `load_intervention_scenario` method loads additional scenarios for comparison. Each is stored using a user-defined name:

```python
def load_intervention_scenario(self, intervention_path: str, scenario_name: str):
    print(f"Loading intervention scenario '{scenario_name}' from: {intervention_path}")
    data, metadata = self.load_raster(intervention_path)
    self.intervention_data[scenario_name] = data
    print(f"Intervention '{scenario_name}' data shape: {data.shape}")
    print(f"Intervention '{scenario_name}' data range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f} {self.service_unit}")
```

When land-use-based valuation is used, the `load_land_use_map` method loads the corresponding raster and displays the unique land-use codes:

```python
def load_land_use_map(self, land_use_path: str):
    print(f"Loading land use map from: {land_use_path}")
    self.land_use_map, self.land_use_metadata = self.load_raster(land_use_path)
    print(f"Land use map shape: {self.land_use_map.shape}")
    print(f"Land use map unique values: {np.unique(self.land_use_map)}")
```

These methods ensure that all spatial inputs are consistently loaded and ready for pixel-level analysis.


### Calculation of Delta Service Provision

The `calculate_delta_service_provision` method computes the pixel-level difference between an intervention scenario and the baseline. It uses `numpy` for efficient array operations. The method also prints basic summary statistics to help interpret the spatial effect of the intervention.

```python
def calculate_delta_service_provision(self, scenario_name: str):
    if self.baseline_data is None:
        raise ValueError("Baseline scenario must be loaded first.")
    
    if scenario_name not in self.intervention_data:
        raise ValueError(f"Intervention scenario '{scenario_name}' not found.")
    
    delta_s = self.intervention_data[scenario_name] - self.baseline_data
    self.delta_service_provision[scenario_name] = delta_s
    
    total_delta = np.nansum(delta_s)
    positive_delta = np.nansum(delta_s[delta_s > 0])
    negative_delta = np.nansum(delta_s[delta_s < 0])
    
    print(f"Delta service provision for '{scenario_name}':")
    print(f"  Total change: {total_delta:.2f} {self.service_unit}")
    print(f"  Positive change (increased provision): {positive_delta:.2f} {self.service_unit}")
    print(f"  Negative change (decreased provision): {negative_delta:.2f} {self.service_unit}")
```

The output allows users to quickly assess whether the intervention improved service provision overall, and how gains and losses are distributed. The resulting difference map is stored for use in later valuation steps.


### Monetary Value Mapping

The `calculate_monetary_value_map` method converts changes in service provision into monetary values. It supports two modes: a single global conversion factor or a dictionary of land-use-specific values. If land-use-specific factors are used, the method applies each factor only to pixels with the corresponding land-use code.

```python
def calculate_monetary_value_map(self, scenario_name: str, 
                                 conversion_factors: Union[float, Dict[int, float]]):
    if scenario_name not in self.delta_service_provision:
        raise ValueError(f"Delta service provision for scenario '{scenario_name}' not calculated yet.")
    
    delta_s = self.delta_service_provision[scenario_name]
    monetary_value = np.zeros_like(delta_s, dtype=float)

    if isinstance(conversion_factors, (int, float)):
        monetary_value = delta_s * conversion_factors
        print(f"Applying global conversion factor: {conversion_factors} €/{self.service_unit}")
    elif isinstance(conversion_factors, dict):
        if self.land_use_map is None:
            raise ValueError("Land use map must be loaded for land-use specific conversion factors.")
        
        print("Applying land-use specific conversion factors.")
        for lu_code, factor in conversion_factors.items():
            mask = (self.land_use_map == lu_code) & (~np.isnan(delta_s))
            monetary_value[mask] = delta_s[mask] * factor
            print(f"  Land Use {lu_code}: {factor} €/{self.service_unit}")
    else:
        raise TypeError("conversion_factors must be a float or a dictionary.")

    self.monetary_value_maps[scenario_name] = monetary_value
    
    total_monetary_value = np.nansum(monetary_value)
    print(f"Monetary value map for '{scenario_name}':")
    print(f"  Total monetary value: €{total_monetary_value:.2f}")
```

This method allows users to apply spatially uniform or differentiated valuation strategies. The resulting map is stored and used in financial analysis.


### Financial Metrics Calculation

The `calculate_financial_metrics` method computes the Benefit-Cost Ratio (BCR), Net Present Value (NPV), and Payback Period. It accounts for initial investment costs, annual maintenance costs, project lifespan, and the discount rate. The NPV calculation iteratively discounts annual net benefits over the project lifespan.

```python
    def calculate_financial_metrics(self, scenario_name: str, investment_cost: float, 
                                  project_lifespan: int = 20, annual_maintenance_cost: float = 0.0):
        if scenario_name not in self.monetary_value_maps:
            raise ValueError(f"Monetary value map for scenario '{scenario_name}' not calculated yet.")
        
        total_monetary_value = np.nansum(self.monetary_value_maps[scenario_name])
        
        bcr = total_monetary_value / investment_cost if investment_cost > 0 else float('inf')
        
        npv = 0
        for year in range(project_lifespan):
            annual_net_benefit = total_monetary_value - annual_maintenance_cost
            if year == 0:
                annual_net_benefit -= investment_cost
            npv += annual_net_benefit / ((1 + self.discount_rate) ** year)
        
        payback_period = investment_cost / total_monetary_value if total_monetary_value > 0 else float('inf')
        
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
```

### Visualization and Reporting

The workflow includes methods for visualizing the monetary value maps and comparing financial metrics across scenarios. `matplotlib` and `seaborn` are used to generate high-quality plots. The `visualize_monetary_value_map` method creates heatmaps of the spatially distributed monetary values, while `compare_scenarios` generates bar charts for BCR, NPV, and Payback Period.

```python
    def visualize_monetary_value_map(self, scenario_name: str, save_path: Optional[str] = None):
        # ... (implementation as in the code, creates heatmap)

    def compare_scenarios(self, save_path: Optional[str] = None):
        # ... (implementation as in the code, creates bar charts)
```

Finally, `generate_summary_report` creates a pandas DataFrame summarizing all financial metrics, which can then be saved to a CSV file using `save_results`. This ensures that all key outputs are systematically organized and stored.

```python
    def generate_summary_report(self) -> pd.DataFrame:
        # ... (implementation as in the code, creates pandas DataFrame)

    def save_results(self, output_dir: str):
        # ... (implementation as in the code, saves maps, charts, and CSV)
```
Here is an improved version of your **"Sample Data Generation"** and **"Testing and Validation"** sections. The revision improves clarity, removes unnecessary qualifiers, and uses a consistent, concise style:

---

### Sample Data Generation

A `create_sample_data` function is included for demonstration and testing. It generates synthetic raster datasets for the baseline, one or more intervention scenarios, and a land-use map. The function uses `numpy` to create sample arrays and `rasterio` to write them as GeoTIFF files.

```python
def create_sample_data():
    # ... (implementation creates and saves example GeoTIFFs)
```

This allows users to run and explore the `fcalc` workflow without external model inputs. It also supports unit testing and instructional use.

---

## Testing and Validation

Validation is essential to ensure accuracy and reliability. The `fcalc` tool includes a test suite built with Python’s `unittest` framework. Tests cover individual functions, scenario workflows, and edge cases. The suite checks input handling, numerical correctness, consistency of outputs, and robustness across service types and data formats.

Automated tests help identify implementation errors and confirm expected behavior under a range of conditions. They also support future extensions of the tool.


### Unit Testing Framework

The `unittest` module provides a structured approach to creating test cases. The `TestEcosystemServiceValuationWorkflow` class inherits from `unittest.TestCase` and contains multiple test methods, each designed to verify a specific piece of functionality. The `setUp` method is used to create temporary test data (raster files) before each test, and the `tearDown` method cleans up these temporary files, ensuring a clean testing environment.

```python
import unittest
import tempfile
import os
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np

from ecosystem_service_valuation_workflow import EcosystemServiceValuationWorkflow

class TestEcosystemServiceValuationWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = EcosystemServiceValuationWorkflow(discount_rate=0.03)
        self.temp_dir = tempfile.mkdtemp()
        # ... (rest of setUp method for creating test rasters)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
```


### Key Test Categories and Coverage

1. **Initialization**
   Tests verify that the `EcosystemServiceValuationWorkflow` class initializes correctly with default and custom parameters, such as `discount_rate`.

2. **Raster Loading**
   Tests ensure that raster files are loaded properly. They check data arrays, metadata, and correct assignment of `service_unit`. Errors are tested for missing files or invalid input.

3. **Delta Service Provision Calculation**
   These tests confirm that pixel-wise differences between intervention and baseline rasters are correctly computed. Summary statistics (total, positive, negative) are also verified. Missing baseline or scenario inputs trigger appropriate errors.

4. **Monetary Value Mapping**
   Tests check both global and land-use-specific valuation:

   * Global conversion factor is applied consistently across all pixels.
   * Land-use-specific factors are correctly matched to raster values.
   * Errors are raised if a land-use map is missing when required.

5. **Financial Metrics Calculation**
   This group validates the implementation of Benefit-Cost Ratio (BCR), Net Present Value (NPV), and Payback Period. Tests confirm mathematical correctness and expected outputs under defined input values. Edge cases, such as zero costs or benefits, are also included.

6. **Edge Cases and Consistency**

   * Tests for zero or negative service changes confirm the workflow handles such conditions without failure.
   * Consistency checks confirm that monetary values match between intermediate rasters and final financial outputs.

---

## Results and Analysis

To demonstrate the capabilities of `fcalc`, a synthetic dataset was generated to simulate typical ecosystem service model outputs and land-use classifications. This example shows how the workflow processes scenario data, applies land-use-specific valuation, and computes financial metrics.

### Demonstration Data and Scenarios

The `create_sample_data` function generated five GeoTIFF files:

* `sample_baseline_service_provision.tif`: Baseline ecosystem service values on a 100×100 grid, with values ranging from 10 to 50.
* `sample_intervention_A_service_provision.tif`: Moderate improvement over baseline.
* `sample_intervention_B_service_provision.tif`: Strong overall improvement.
* `sample_intervention_C_service_provision.tif`: Targeted, localized improvements.
* `sample_land_use_map.tif`: Land-use map with four classes (codes 1 to 4) randomly distributed.

The unit of service was defined generically as “units.” In real applications, this could represent physical quantities such as m³ of water or tons of soil.

Each intervention scenario was compared to the baseline. Financial calculations used the following assumptions:

* Discount rate: 3%
* Project lifespan: 20 years
* Investment cost: user-defined per scenario
* Annual maintenance: 1.5% of investment

Valuation used land-use-specific conversion rates (in €/unit of service):

* Land Use 1: 3.0
* Land Use 2: 7.0
* Land Use 3: 4.5
* Land Use 4: 6.0

This setup illustrates the tool’s ability to apply spatially explicit valuation and differentiate economic outcomes by land context. Each intervention produced a distinct monetary value map and associated financial metrics (BCR, NPV, Payback Period), enabling comparative analysis across scenarios.


### Financial Outcomes and Comparative Analysis

To test the `fcalc` workflow, three intervention scenarios were evaluated using synthetic data. The inputs, including service provision rasters and land-use classifications, were randomly generated and do not reflect real environmental conditions. The purpose of this analysis is to demonstrate the tool’s computational logic and financial assessment capabilities.

The output below summarizes the financial metrics calculated for each scenario:

```
      Scenario  Total Monetary Value (€)  Investment Cost (€)   BCR     NPV (€)     Payback Period (years)
intervention_A           514,609.73              75,000.00      6.86   7,793,536.92            0.1
intervention_B           894,608.25             120,000.00      7.46  13,561,214.22            0.1
intervention_C           331,646.28              60,000.00      5.53   5,008,289.55            0.2
```

These figures reflect the synthetic structure of the input data and valuation parameters. The total monetary value represents the sum of pixel-wise avoided costs over a 20-year project lifespan. The Benefit-Cost Ratio (BCR), Net Present Value (NPV), and Payback Period are calculated using a 3% discount rate and assumed maintenance costs (1.5% of investment per year).

Key outputs include:

* All scenarios produced high BCRs, with intervention B showing the largest return.
* NPVs were positive across all scenarios, driven by large synthetic benefit values.
* Payback periods were unrealistically short due to the simplified data, highlighting how input assumptions affect results.

These results are for illustrative purposes only. They show how `fcalc` can process spatial data, apply valuation rules, and generate comparative financial metrics. A real-world application would require calibrated input data, validated unit values, and context-specific cost structures.

A visualization (`scenario_comparison.png`) was also produced to compare financial metrics across scenarios. It supports the interpretation of results but should be viewed as a demonstration, not as decision-support evidence.


### Spatial Distribution of Monetary Value

In addition to summary metrics, `fcalc` generates spatial maps showing the distribution of monetary value across the study area. These outputs (e.g., `monetary_value_map_intervention_A.png`, etc.) visualize where each intervention yields higher or lower economic returns at the pixel level. Warmer colors typically represent higher values.

These maps are useful for spatial analysis and planning. They can help identify locations where interventions provide greater economic benefit, such as areas dominated by land-use classes with higher conversion factors. This supports more targeted investment and prioritization of resources.


### Interpretation and Applicability

The demonstration confirms that the workflow can perform a complete ecosystem service valuation using synthetic inputs. It shows how the system:

* Converts service change into monetary terms.
* Compares multiple scenarios using financial metrics.
* Maps the spatial pattern of benefits.
* Applies land-use-specific valuation rules.

Although the data used here are artificial, the methods are directly transferable to real-world applications. With appropriate inputs—such as outputs from InVEST or other models—`fcalc` can be applied to value services like flood mitigation, erosion control, or water purification by adjusting service units, input rasters, and valuation factors.

---

## Usage Guidelines

This section outlines how to use the `fcalc` ecosystem service valuation tool, including setup requirements, input data, execution steps, and interpretation of outputs. The workflow is designed to support a range of ecosystem services and valuation methods.

### Prerequisites and Installation

`fcalc` is implemented in Python and requires version 3.8 or higher. It depends on several commonly used libraries for geospatial processing and analysis:

* `numpy`: numerical operations and array handling
* `rasterio`: reading and writing raster (GeoTIFF) files
* `matplotlib`: basic data visualization
* `seaborn`: enhanced plotting and statistical visuals
* `pandas`: tabular data manipulation and reporting

To install the required packages, run:

```bash
pip install numpy rasterio matplotlib seaborn pandas
```

These dependencies work on all major platforms (Windows, macOS, Linux). Users should verify that all packages are installed before running the workflow. No additional software or external dependencies are required.


### Input Data Requirements

The workflow requires geospatial raster data as input, typically in GeoTIFF format. The primary inputs are:

1.  **Baseline Ecosystem Service Raster:** A single-band raster file representing the provision of the ecosystem service under current or pre-intervention conditions. This serves as the reference against which changes are measured. Example filename: `baseline_service_provision.tif`.

2.  **Intervention Ecosystem Service Rasters:** One or more single-band raster files, each representing the provision of the ecosystem service after a specific intervention (e.g., implementation of green infrastructure, land-use change, restoration project). Each intervention scenario should have its own raster file. Example filenames: `intervention_A_service_provision.tif`, `intervention_B_service_provision.tif`.

3.  **Land Use Map Raster (Optional):** A single-band raster file representing the land cover or land use types of the study area. This is required only if you intend to use land-use specific conversion factors for monetary valuation. The pixel values in this raster should correspond to distinct land-use codes (e.g., integers representing different land cover classes). Example filename: `land_use_map.tif`.

**Important Considerations for Input Rasters:**

*   **Spatial Consistency:** All input rasters (baseline, intervention, and land-use map) **must** share consistent spatial characteristics. This includes having the same Coordinate Reference System (CRS), spatial extent (bounding box), and pixel resolution. The workflow includes internal checks for spatial consistency, but ensuring this beforehand will prevent errors.
*   **Data Type:** The ecosystem service rasters should contain quantitative values representing the service provision (e.g., cubic meters, tons, kilograms, or abstract units). The land-use map typically contains integer codes.
*   **NoData Values:** Ensure that NoData values are properly defined in the raster metadata, as the workflow uses `np.nansum` and `np.nanmin`/`np.nanmax` to handle missing data or areas outside the study extent.


### Workflow Execution Process

To use the workflow, you will typically follow these steps within a Python script:

1.  **Import the Workflow Class:**
    ```python
    from ecosystem_service_valuation_workflow import EcosystemServiceValuationWorkflow
    ```

2.  **Initialize the Workflow:** Create an instance of the `EcosystemServiceValuationWorkflow` class. You can specify a `discount_rate` (default is 0.03 for 3%).
    ```python
    workflow = EcosystemServiceValuationWorkflow(discount_rate=0.03)
    ```

3.  **Load Baseline Scenario:** Load your baseline ecosystem service raster. You must also specify the `service_unit` (e.g., "m³", "tons", "kg", "units"). This unit will be used in all subsequent print statements and visualizations.
    ```python
    workflow.load_baseline_scenario("path/to/your/baseline_service.tif", service_unit="m³")
    ```

4.  **Load Intervention Scenarios:** Load each of your intervention ecosystem service rasters, providing a unique `scenario_name` for each.
    ```python
    workflow.load_intervention_scenario("path/to/your/intervention_A_service.tif", "Intervention_A")
    workflow.load_intervention_scenario("path/to/your/intervention_B_service.tif", "Intervention_B")
    # Add more as needed
    ```

5.  **Load Land Use Map (Optional):** If you plan to use land-use specific conversion factors, load your land-use map.
    ```python
    workflow.load_land_use_map("path/to/your/land_use_map.tif")
    ```

6.  **Define Conversion Factors:** Decide whether to use a global conversion factor or land-use specific factors.
    *   **Global Factor:** A single float value (e.g., `global_factor = 10.0` for 10 €/unit).
    *   **Land-Use Specific Factors:** A dictionary mapping land-use codes (integers from your land-use map) to their respective conversion factors (e.g., `lu_factors = {1: 5.0, 2: 12.0, 3: 8.0}`).

7.  **Define Investment Costs:** Create a dictionary mapping your scenario names to their total initial investment costs (in €).
    ```python
    investment_costs = {
        "Intervention_A": 100000.0,
        "Intervention_B": 150000.0
    }
    ```

8.  **Process Each Intervention Scenario:** Iterate through your intervention scenarios to perform the calculations.
    ```python
    for scenario_name in workflow.intervention_data.keys():
        print(f"\n--- Processing {scenario_name} ---")
        
        # Calculate change in service provision
        workflow.calculate_delta_service_provision(scenario_name)
        
        # Calculate monetary value map (choose one of the following)
        workflow.calculate_monetary_value_map(scenario_name, global_factor) # Using global factor
        # OR
        # workflow.calculate_monetary_value_map(scenario_name, lu_factors) # Using land-use specific factors
        
        # Calculate financial metrics
        workflow.calculate_financial_metrics(
            scenario_name, 
            investment_costs[scenario_name], 
            project_lifespan=20, # Example: 20 years
            annual_maintenance_cost=investment_costs[scenario_name] * 0.01 # Example: 1% annual maintenance
        )
    ```

9.  **Save Results:** Save all generated maps, charts, and the financial summary report to a specified output directory.
    ```python
    workflow.save_results("my_valuation_results")
    ```

10. **View Summary Report:** Optionally, print the financial summary report to the console.
    ```python
    summary_df = workflow.generate_summary_report()
    print("\n--- Financial Summary Report ---")
    print(summary_df.to_string(index=False))
    ```

### Parameter Configuration Guidelines

*   **`discount_rate`**: This parameter reflects the time preference for money and should align with standard economic analysis practices in your jurisdiction. Government agencies typically use discount rates between 2% and 7%, while private sector analyses may employ higher rates reflecting risk premiums and opportunity costs.

*   **`service_unit`**: Clearly define the unit of the ecosystem service (e.g., "m³", "tons", "kg", "units"). This ensures clarity in outputs and consistency in calculations.

*   **`conversion_factors`**: This is a critical parameter. Its accuracy directly impacts the monetary valuation. 
    *   **Global Factor**: A single value representing the economic value per unit of service. This could be derived from benefit transfer studies, market prices (if applicable), or avoided cost estimates (e.g., cost of treating water if natural purification is lost).
    *   **Land-Use Specific Factors**: A dictionary where keys are integer land-use codes and values are the economic value per unit of service for that specific land-use type. These values should be based on robust economic studies or expert elicitation, reflecting the varying economic impact of service changes across different land covers.

*   **`investment_cost`**: This should be the total initial capital expenditure required for implementing the intervention. It should include all relevant expenses associated with the project, such as planning, design, materials, labor, and initial establishment costs.

*   **`project_lifespan`**: Represents the expected useful life of the intervention in years. This parameter is crucial for NPV calculations. Typical values range from 15 to 50 years, depending on the type of intervention (e.g., tree planting vs. engineered green infrastructure).

*   **`annual_maintenance_cost`**: Any recurring costs associated with maintaining the intervention over its lifespan. This could include operational costs, repairs, or ongoing management expenses. If not applicable, set to 0.0.


### Output Interpretation

The workflow generates several outputs that provide a comprehensive view of the economic valuation:

*   **Monetary Value Maps (PNG files):** These spatially explicit maps illustrate where the economic benefits (or costs) are concentrated. Areas with higher values indicate locations where the intervention yields greater economic returns. These maps are invaluable for spatial planning and identifying high-impact areas.

*   **Scenario Comparison Chart (PNG file):** This chart visually compares the BCR, NPV, and Payback Period across all evaluated intervention scenarios. It provides a quick and intuitive way to assess the relative financial attractiveness of different options.

*   **Financial Summary Report (CSV file):** A tabular summary of all calculated financial metrics for each scenario. This report is suitable for detailed analysis and integration into other documents or spreadsheets.

When interpreting the results, consider the following:

*   **BCR > 1.0 and Positive NPV:** Indicate economically viable interventions where benefits outweigh costs.
*   **Shorter Payback Period:** Generally preferred, as it signifies a quicker return on investment.
*   **Spatial Patterns in Monetary Value Maps:** Use these to understand the distribution of benefits and to inform targeted implementation strategies.


### Customization and Extension

The `EcosystemServiceValuationWorkflow` class is modular and can be adapted for different use cases:

* **New Ecosystem Services**: To evaluate a different service, provide the baseline and intervention rasters, set the appropriate `service_unit`, and define `conversion_factors` (global or land-use-specific).
* **Valuation Methods**: The current implementation uses the avoided cost method. Other methods can be added by modifying or extending the `calculate_monetary_value_map` function.
* **Financial Metrics**: Additional indicators, such as Internal Rate of Return (IRR), can be implemented by updating the `calculate_financial_metrics` method.
* **Visualizations**: Custom charts or map outputs can be added by modifying the reporting functions.

This structure allows users to tailor the tool to specific services, regions, or analytical needs, supporting a range of ecosystem valuation applications.

---

## Conclusions and Recommendations

### Implementation and Validation

The `fcalc` tool has been implemented, tested, and validated. It provides a consistent method for estimating the economic value of ecosystem service interventions. Originally developed for flood risk mitigation, the tool has been generalized to support a wide range of services and valuation contexts.

The system is modular and follows an object-oriented structure. It supports global and land-use-specific valuation and can process standard geospatial model outputs. A complete test suite was developed, confirming the correctness of each component—from data loading to financial metric calculation.

The demonstration using synthetic data showed that the tool can compare multiple scenarios, apply spatially explicit valuation, and generate financial indicators. It meets the initial goal of supporting economic assessments across services like soil retention, water quality, and carbon storage. By separating core logic from service-specific inputs, `fcalc` enables a repeatable and scalable approach to ecosystem service valuation.


Here is a revised version of the **“Scientific and Methodological Contributions”** and **“Policy and Planning Implications”** sections. The text has been edited for neutrality, clarity, and brevity, while preserving the intended meaning:

---

### Scientific and Methodological Contributions

This work contributes to ecosystem service valuation by:

* **Supporting multiple services**: The tool is not limited to one type of ecosystem service. It can be applied to various models and services, improving consistency in natural capital accounting.
* **Accounting for spatial variation**: Land-use-specific conversion factors allow for more detailed economic valuation by reflecting differences across landscapes.
* **Ensuring transparency**: The implementation includes clear code and a full test suite. This supports reproducibility and independent validation.
* **Linking ecology to finance**: The tool converts biophysical outputs into financial metrics such as BCR, NPV, and Payback Period. These indicators are accessible to non-specialists and relevant to policy.


### Policy and Planning Implications

The availability of a generic valuation tool can support decision-making in several ways:

* **Investment prioritization**: The tool helps identify interventions with the highest economic return, improving the use of public or private funds.
* **Integration of natural capital**: It facilitates the inclusion of ecosystem values in infrastructure planning and development policy.
* **Cross-sector alignment**: A unified valuation method allows actors from different sectors to use shared indicators for evaluating environmental actions.
* **Monitoring and evaluation**: Results can be used to track changes over time and evaluate the impact of nature-based solutions or land management interventions.


### Technical Recommendations

While the current implementation meets core functional goals, several areas may be considered for future development:

* **User Interface**: A graphical or web-based interface could improve accessibility for non-technical users by simplifying data input and visualization.
* **GIS Integration**: Connecting the tool to GIS platforms such as QGIS or ArcGIS would streamline data preparation and support workflows used by spatial analysts.
* **Uncertainty Analysis**: A module for sensitivity and uncertainty analysis (e.g., Monte Carlo simulation) could help assess how results vary with input assumptions.
* **Additional Financial Metrics**: Including metrics such as Internal Rate of Return (IRR) or annualized net benefits would allow more detailed financial analysis.
* **Dynamic Valuation**: Methods that account for temporal changes in ecosystem services, such as climate or land-use change, could improve long-term assessments.


### Research and Development Priorities

Future research could focus on improving data quality, analytical scope, and decision relevance:

* **Empirical Valuation Data**: Collecting context-specific data to improve the accuracy of unit conversion factors across services and land-use types.
* **Model Output Standardization**: Promoting standard formats and units across ecosystem service models to support easier integration into valuation tools.
* **Co-benefits and Trade-offs**: Expanding the tool to quantify interactions among multiple services would support more holistic evaluation of interventions.
* **Socio-Economic Integration**: Adding layers such as population, income, or vulnerability indices would refine spatial prioritization and better align valuation with human outcomes.


### Scaling and Adoption Strategies

To support broader adoption of the `fcalc` tool, the following actions are recommended:

* **Training and Capacity Building**: Develop tutorials, documentation, and workshops to support users from public institutions, NGOs, and consulting groups.
* **Case Study Library**: Provide examples of the tool applied to various ecosystem services and regions to demonstrate its practical relevance.
* **Open-Source Engagement**: Encourage community contributions by hosting the code in an open repository and inviting feedback, extensions, and improvements.
* **Policy Integration**: Work with policymakers to promote ecosystem service valuation as part of planning, regulatory, and investment frameworks.

### Long-term Vision and Impact

The goal is to establish `fcalc` as a standard tool for incorporating ecosystem service values into environmental and economic decision-making. By offering a flexible and transparent method for spatial valuation, the tool can support a shift toward planning approaches that recognize and account for the role of natural systems. This contributes to long-term strategies for sustainable land use, climate adaptation, and investment in nature-based solutions.

