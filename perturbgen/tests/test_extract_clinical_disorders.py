"""
Unit tests for the extract_clinical_disorders utility function.
"""
import unittest
import pandas as pd
import numpy as np

from perturbgen.src.utils import extract_clinical_disorders


class TestExtractClinicalDisorders(unittest.TestCase):
    """Test cases for extract_clinical_disorders function."""
    
    def test_basic_extraction(self):
        """Test basic extraction of clinical disorders."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia', 'Thrombocytopenia'],
            'Severity': ['Mild', 'Severe', 'Moderate']
        })
        result = extract_clinical_disorders(disease_df)
        
        self.assertIsInstance(result, str)
        self.assertIn('Anemia', result)
        self.assertIn('Leukemia', result)
        self.assertIn('Thrombocytopenia', result)
    
    def test_filtered_extraction(self):
        """Test extraction with filter applied."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia', 'Thrombocytopenia'],
            'Severity': ['Mild', 'Severe', 'Moderate']
        })
        result = extract_clinical_disorders(
            disease_df,
            filter_column='Severity',
            filter_value='Severe'
        )
        
        self.assertEqual(result, 'Leukemia')
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = extract_clinical_disorders(empty_df)
        
        self.assertEqual(result, "")
    
    def test_none_dataframe(self):
        """Test handling of None DataFrame."""
        result = extract_clinical_disorders(None)
        
        self.assertEqual(result, "")
    
    def test_duplicate_disorders(self):
        """Test that duplicate disorders are handled correctly (unique values only)."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia', 'Anemia', 'Thrombocytopenia'],
            'Type': ['Type1', 'Type2', 'Type3', 'Type4']
        })
        result = extract_clinical_disorders(disease_df)
        
        # Count occurrences of 'Anemia' - should only appear once
        anemia_count = result.count('Anemia')
        self.assertEqual(anemia_count, 1)
    
    def test_nan_values(self):
        """Test that NaN values are properly excluded."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', None, 'Leukemia', pd.NA],
            'Type': ['Type1', 'Type2', 'Type3', 'Type4']
        })
        result = extract_clinical_disorders(disease_df)
        
        self.assertNotIn('None', result)
        self.assertNotIn('nan', result.lower())
        self.assertIn('Anemia', result)
        self.assertIn('Leukemia', result)
    
    def test_custom_column_name(self):
        """Test extraction with custom column name."""
        disease_df = pd.DataFrame({
            'Disease': ['Diabetes', 'Hypertension'],
            'Status': ['Active', 'Resolved']
        })
        result = extract_clinical_disorders(disease_df, column_name='Disease')
        
        self.assertIn('Diabetes', result)
        self.assertIn('Hypertension', result)
    
    def test_invalid_column_name(self):
        """Test that ValueError is raised for invalid column name."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia'],
            'Severity': ['Mild', 'Severe']
        })
        
        with self.assertRaises(ValueError):
            extract_clinical_disorders(disease_df, column_name='NonExistentColumn')
    
    def test_invalid_filter_column(self):
        """Test that ValueError is raised for invalid filter column."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia'],
            'Severity': ['Mild', 'Severe']
        })
        
        with self.assertRaises(ValueError):
            extract_clinical_disorders(
                disease_df,
                filter_column='NonExistentColumn',
                filter_value='SomeValue'
            )
    
    def test_multiple_matching_filters(self):
        """Test extraction with multiple rows matching filter."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia', 'Neutropenia', 'Thrombocytopenia'],
            'Category': ['Blood', 'Blood', 'Blood', 'Platelet']
        })
        result = extract_clinical_disorders(
            disease_df,
            filter_column='Category',
            filter_value='Blood'
        )
        
        self.assertIn('Anemia', result)
        self.assertIn('Leukemia', result)
        self.assertIn('Neutropenia', result)
        self.assertNotIn('Thrombocytopenia', result)
    
    def test_no_matching_filter(self):
        """Test extraction when filter matches no rows."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia'],
            'Severity': ['Mild', 'Severe']
        })
        result = extract_clinical_disorders(
            disease_df,
            filter_column='Severity',
            filter_value='Critical'
        )
        
        self.assertEqual(result, "")
    
    def test_comma_separation(self):
        """Test that output is properly comma-separated."""
        disease_df = pd.DataFrame({
            'Clinical disorder': ['Anemia', 'Leukemia', 'Thrombocytopenia']
        })
        result = extract_clinical_disorders(disease_df)
        
        # Check that commas are present
        self.assertIn(',', result)
        # Check that splitting by comma gives correct number of items
        items = [item.strip() for item in result.split(',')]
        self.assertEqual(len(items), 3)


if __name__ == '__main__':
    unittest.main()
