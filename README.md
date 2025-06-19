# Financial Calculator for Ecosystem Services

This repository provides a Python-based tool for valuing ecosystem services using spatial model outputs. It supports unit-to-monetary conversion and integrates land-use specific valuation. The tool is applicable to services such as flood risk mitigation, soil erosion prevention, water purification, and carbon storage.

## Features

* **Generic Input Support**: Accepts raster outputs from different ecosystem service models.
* **Flexible Valuation**: Supports global and land-use specific conversion factors.
* **Financial Indicators**: Computes Benefit-Cost Ratio (BCR), Net Present Value (NPV), and Payback Period.
* **Spatial Outputs**: Produces monetary value maps to visualize benefit distribution.
* **Scenario Comparison**: Generates charts to compare financial results across interventions.
* **Documentation**: Includes full technical reference, usage instructions, and examples.

## Repository Structure

```
generic_es_valuation_workflow_repo/
├── README.md
├── src/
│   ├── ecosystem_service_valuation_workflow.py
│   └── test_ecosystem_service_workflow.py
├── docs/
│   ├── Generic_ES_Valuation_Documentation.md
│   └── Generic_ES_Valuation_Documentation.pdf
├── data/
│   ├── sample_baseline_service_provision.tif
│   ├── sample_intervention_A_service_provision.tif
│   ├── sample_intervention_B_service_provision.tif
│   ├── sample_intervention_C_service_provision.tif
│   └── sample_land_use_map.tif
├── results/
│   ├── ecosystem_service_results/
│   │   ├── monetary_value_map_intervention_A.png
│   │   ├── monetary_value_map_intervention_B.png
│   │   ├── monetary_value_map_intervention_C.png
│   │   ├── scenario_comparison.png
│   │   └── financial_summary_report.csv
└── .gitignore
```

## Getting Started

### Prerequisites

-   Python 3.8+
-   `numpy`
-   `rasterio`
-   `matplotlib`
-   `seaborn`
-   `pandas`

Install dependencies using pip:

```bash
pip install numpy rasterio matplotlib seaborn pandas
```

### Running the Workflow

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Open-Earth-Monitor/fcalc.git
    cd fcalc
    ```

2.  **Place your data:** If you have your own ecosystem service model outputs (baseline and intervention rasters) and an optional land-use map, place them in the `data/` directory. You can also use the provided sample data.

3.  **Configure and Run:** Open `src/ecosystem_service_valuation_workflow.py` and modify the `if __name__ == "__main__":` block to configure your specific scenarios, investment costs, and conversion factors. Then, run the script:
    ```bash
    python src/ecosystem_service_valuation_workflow.py
    ```

4.  **View Results:** Outputs (monetary value maps, comparison charts, and a financial summary CSV) will be saved in the `results/ecosystem_service_results/` directory.

### Running Tests

To run the comprehensive test suite and validate the workflow:

```bash
python src/test_ecosystem_service_workflow.py
```

## Documentation

Detailed documentation, including methodology, implementation details, usage guidelines, and results analysis, can be found in the `docs/` directory:

-   `Generic_ES_Valuation_Documentation.md` (Markdown format)

## License

This project is licensed under the MIT License - see the LICENSE file for details. *(Note: A LICENSE file is not included in this generation but is recommended for a real repository.)*

## Contact

For questions or collaborations, please contact [Your Name/Email/GitHub Profile].

