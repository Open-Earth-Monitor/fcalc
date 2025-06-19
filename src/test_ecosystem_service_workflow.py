#!/usr/bin/env python3
"""
Test Suite for Generic Ecosystem Service Valuation Workflow

This module contains comprehensive tests to validate the generalized ecosystem service valuation workflow implementation.
It tests various components including data loading, calculations, and edge cases, with a focus on generic applicability.

Author: Manus AI
Date: June 19, 2025
"""

import numpy as np
import unittest
import tempfile
import os
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Import the generalized workflow class
from ecosystem_service_valuation_workflow import EcosystemServiceValuationWorkflow


class TestEcosystemServiceValuationWorkflow(unittest.TestCase):
    """
    Test cases for the Generic Ecosystem Service Valuation Workflow.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.workflow = EcosystemServiceValuationWorkflow(discount_rate=0.03)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test raster data
        self.height, self.width = 50, 50
        self.transform = from_bounds(0, 0, self.width, self.height, self.width, self.height)
        self.crs = CRS.from_epsg(4326)
        
        # Create baseline and intervention test data
        np.random.seed(123)  # For reproducible tests
        self.baseline_data = np.random.uniform(10, 50, (self.height, self.width))
        self.intervention_data = self.baseline_data + np.random.uniform(5, 15, (self.height, self.width))
        
        # Create land use map data
        self.land_use_data = np.random.randint(1, 5, (self.height, self.width)) # 4 land use types
        
        # Create test raster files
        self.baseline_path = os.path.join(self.temp_dir, "test_baseline_service.tif")
        self.intervention_path = os.path.join(self.temp_dir, "test_intervention_service.tif")
        self.land_use_path = os.path.join(self.temp_dir, "test_land_use.tif")
        
        self._create_test_raster(self.baseline_path, self.baseline_data)
        self._create_test_raster(self.intervention_path, self.intervention_data)
        self._create_test_raster(self.land_use_path, self.land_use_data, dtype=rasterio.uint8) # Land use typically uint
    
    def tearDown(self):
        """
        Clean up after each test method.
        """
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_raster(self, filepath, data, dtype=None):
        """
        Helper method to create a test raster file.
        """
        if dtype is None:
            dtype = data.dtype
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=1,
            dtype=dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            dst.write(data, 1)
    
    def test_initialization(self):
        """
        Test workflow initialization with default and custom parameters.
        """
        workflow_default = EcosystemServiceValuationWorkflow()
        self.assertEqual(workflow_default.discount_rate, 0.03)
        
        workflow_custom = EcosystemServiceValuationWorkflow(discount_rate=0.05)
        self.assertEqual(workflow_custom.discount_rate, 0.05)
    
    def test_load_raster(self):
        """
        Test raster loading functionality.
        """
        data, metadata = self.workflow.load_raster(self.baseline_path)
        
        self.assertEqual(data.shape, (self.height, self.width))
        self.assertTrue(isinstance(data, np.ndarray))
        
        self.assertIn('transform', metadata)
        self.assertIn('crs', metadata)
        
        with self.assertRaises(FileNotFoundError):
            self.workflow.load_raster("nonexistent_file.tif")
    
    def test_load_baseline_scenario(self):
        """
        Test loading baseline scenario.
        """
        self.workflow.load_baseline_scenario(self.baseline_path, service_unit="tons")
        
        self.assertIsNotNone(self.workflow.baseline_data)
        self.assertEqual(self.workflow.baseline_data.shape, (self.height, self.width))
        self.assertEqual(self.workflow.service_unit, "tons")
    
    def test_load_intervention_scenario(self):
        """
        Test loading intervention scenario.
        """
        scenario_name = "test_intervention"
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        
        self.assertIn(scenario_name, self.workflow.intervention_data)
        self.assertEqual(self.workflow.intervention_data[scenario_name].shape, (self.height, self.width))
    
    def test_load_land_use_map(self):
        """
        Test loading land use map.
        """
        self.workflow.load_land_use_map(self.land_use_path)
        
        self.assertIsNotNone(self.workflow.land_use_map)
        self.assertEqual(self.workflow.land_use_map.shape, (self.height, self.width))
        self.assertTrue(np.array_equal(np.unique(self.workflow.land_use_map), np.array([1, 2, 3, 4])))

    def test_calculate_delta_service_provision(self):
        """
        Test delta service provision calculation.
        """
        scenario_name = "test_intervention"
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        
        self.workflow.calculate_delta_service_provision(scenario_name)
        
        self.assertIn(scenario_name, self.workflow.delta_service_provision)
        delta_s = self.workflow.delta_service_provision[scenario_name]
        
        self.assertEqual(delta_s.shape, (self.height, self.width))
        self.assertGreater(np.nansum(delta_s), 0) # Intervention should improve service
        
        with self.assertRaises(ValueError):
            workflow_empty = EcosystemServiceValuationWorkflow()
            workflow_empty.calculate_delta_service_provision(scenario_name)
        
        with self.assertRaises(ValueError):
            self.workflow.calculate_delta_service_provision("nonexistent_scenario")
    
    def test_calculate_monetary_value_map_global_factor(self):
        """
        Test monetary value map calculation with a global conversion factor.
        """
        scenario_name = "test_intervention"
        global_factor = 5.0
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        self.workflow.calculate_delta_service_provision(scenario_name)
        
        self.workflow.calculate_monetary_value_map(scenario_name, global_factor)
        
        self.assertIn(scenario_name, self.workflow.monetary_value_maps)
        monetary_value = self.workflow.monetary_value_maps[scenario_name]
        
        self.assertEqual(monetary_value.shape, (self.height, self.width))
        self.assertGreater(np.nansum(monetary_value), 0)
        
        expected_value = self.workflow.delta_service_provision[scenario_name] * global_factor
        np.testing.assert_array_almost_equal(monetary_value, expected_value)
    
    def test_calculate_monetary_value_map_land_use_specific(self):
        """
        Test monetary value map calculation with land-use specific conversion factors.
        """
        scenario_name = "test_intervention"
        land_use_factors = {1: 3.0, 2: 7.0, 3: 4.5, 4: 6.0}
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        self.workflow.load_land_use_map(self.land_use_path)
        self.workflow.calculate_delta_service_provision(scenario_name)
        
        self.workflow.calculate_monetary_value_map(scenario_name, land_use_factors)
        
        self.assertIn(scenario_name, self.workflow.monetary_value_maps)
        monetary_value = self.workflow.monetary_value_maps[scenario_name]
        
        self.assertEqual(monetary_value.shape, (self.height, self.width))
        self.assertGreater(np.nansum(monetary_value), 0)
        
        # Verify some pixel values based on land use
        delta_s = self.workflow.delta_service_provision[scenario_name]
        for r in range(self.height):
            for c in range(self.width):
                lu_code = self.land_use_data[r, c]
                expected_val = delta_s[r, c] * land_use_factors[lu_code]
                self.assertAlmostEqual(monetary_value[r, c], expected_val)
        
        with self.assertRaises(ValueError):
            # Land use map not loaded
            workflow_no_lu = EcosystemServiceValuationWorkflow()
            workflow_no_lu.load_baseline_scenario(self.baseline_path)
            workflow_no_lu.load_intervention_scenario(self.intervention_path, scenario_name)
            workflow_no_lu.calculate_delta_service_provision(scenario_name)
            workflow_no_lu.calculate_monetary_value_map(scenario_name, land_use_factors)
    
    def test_calculate_financial_metrics(self):
        """
        Test financial metrics calculation.
        """
        scenario_name = "test_intervention"
        investment_cost = 100000.0
        project_lifespan = 20
        annual_maintenance_cost = 2000.0
        global_factor = 5.0
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        self.workflow.calculate_delta_service_provision(scenario_name)
        self.workflow.calculate_monetary_value_map(scenario_name, global_factor)
        
        self.workflow.calculate_financial_metrics(
            scenario_name, investment_cost, project_lifespan, annual_maintenance_cost
        )
        
        self.assertIn(scenario_name, self.workflow.financial_metrics)
        metrics = self.workflow.financial_metrics[scenario_name]
        
        required_keys = ['total_monetary_value', 'investment_cost', 'bcr', 'npv', 'payback_period']
        for key in required_keys:
            self.assertIn(key, metrics)
        
        expected_bcr = metrics['total_monetary_value'] / investment_cost
        self.assertAlmostEqual(metrics['bcr'], expected_bcr, places=5)
        self.assertGreater(metrics['bcr'], 0)
        self.assertGreater(metrics['payback_period'], 0)
        
        with self.assertRaises(ValueError):
            self.workflow.calculate_financial_metrics("nonexistent_scenario", investment_cost)
    
    def test_edge_cases(self):
        """
        Test edge cases and boundary conditions.
        """
        scenario_name = "edge_case"
        
        # Test with zero delta service provision
        zero_data = np.copy(self.baseline_data) # Same as baseline
        zero_path = os.path.join(self.temp_dir, "zero_intervention_service.tif")
        self._create_test_raster(zero_path, zero_data)
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(zero_path, scenario_name)
        self.workflow.calculate_delta_service_provision(scenario_name)
        
        delta_sum = np.nansum(self.workflow.delta_service_provision[scenario_name])
        self.assertAlmostEqual(delta_sum, 0, places=1)
        
        # Test with negative delta service provision (intervention worse than baseline)
        worse_data = self.baseline_data - np.random.uniform(1, 5, (self.height, self.width))
        worse_data = np.clip(worse_data, 0, None) # Ensure non-negative service values
        worse_path = os.path.join(self.temp_dir, "worse_intervention_service.tif")
        self._create_test_raster(worse_path, worse_data)
        
        scenario_name_worse = "worse_intervention"
        self.workflow.load_intervention_scenario(worse_path, scenario_name_worse)
        self.workflow.calculate_delta_service_provision(scenario_name_worse)
        
        delta_sum_worse = np.nansum(self.workflow.delta_service_provision[scenario_name_worse])
        self.assertLess(delta_sum_worse, 0)
    
    def test_data_consistency(self):
        """
        Test data consistency across different operations.
        """
        scenario_name = "consistency_test"
        global_factor = 10.0
        
        self.workflow.load_baseline_scenario(self.baseline_path)
        self.workflow.load_intervention_scenario(self.intervention_path, scenario_name)
        
        self.workflow.calculate_delta_service_provision(scenario_name)
        delta_s = self.workflow.delta_service_provision[scenario_name]
        
        self.workflow.calculate_monetary_value_map(scenario_name, global_factor)
        monetary_value = self.workflow.monetary_value_maps[scenario_name]
        
        expected_monetary_value = delta_s * global_factor
        np.testing.assert_array_almost_equal(monetary_value, expected_monetary_value)
        
        investment_cost = 50000.0
        self.workflow.calculate_financial_metrics(scenario_name, investment_cost)
        
        total_from_map = np.nansum(monetary_value)
        total_from_metrics = self.workflow.financial_metrics[scenario_name]['total_monetary_value']
        self.assertAlmostEqual(total_from_map, total_from_metrics, places=2)


def run_validation_tests():
    """
    Run validation tests and generate a report.
    """
    print("=== Generic Ecosystem Service Valuation Workflow Validation Tests ===\n")
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEcosystemServiceValuationWorkflow)
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


def validate_workflow_outputs():
    """
    Validate the outputs generated by the workflow.
    """
    print("\n=== Validating Workflow Outputs ===\n")
    
    output_dir = Path("ecosystem_service_results")
    if not output_dir.exists():
        print("❌ Output directory 'ecosystem_service_results' not found")
        return False
    
    required_files = [
        "monetary_value_map_intervention_A.png",
        "monetary_value_map_intervention_B.png", 
        "monetary_value_map_intervention_C.png",
        "scenario_comparison.png",
        "financial_summary_report.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = output_dir / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    csv_path = output_dir / "financial_summary_report.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            required_columns = ['Scenario', 'Total Monetary Value (€)', 'Investment Cost (€)', 'BCR', 'NPV (€)', 'Payback Period (years)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ CSV missing columns: {missing_columns}")
            else:
                print(f"✅ CSV has all required columns")
                print(f"✅ CSV contains {len(df)} scenarios")
                
                for _, row in df.iterrows():
                    scenario = row['Scenario']
                    bcr = float(row['BCR'])
                    if bcr > 0:
                        print(f"✅ {scenario}: BCR = {bcr:.2f} (positive)")
                    else:
                        print(f"⚠️  {scenario}: BCR = {bcr:.2f} (non-positive)")
                        
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
    
    if not missing_files:
        print(f"\n✅ All required output files are present")
        return True
    else:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False


if __name__ == "__main__":
    tests_passed = run_validation_tests()
    outputs_valid = validate_workflow_outputs()
    
    print(f"\n=== Overall Validation Result ===")
    if tests_passed and outputs_valid:
        print("✅ Generic Ecosystem Service Valuation Workflow validation PASSED")
        print("The generalized implementation is working correctly and producing expected outputs.")
    else:
        print("❌ Generic Ecosystem Service Valuation Workflow validation FAILED")
        if not tests_passed:
            print("- Unit tests failed")
        if not outputs_valid:
            print("- Output validation failed")


