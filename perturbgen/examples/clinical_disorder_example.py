"""
Example: Extracting Clinical Disorder Information from DataFrames

This example demonstrates how to use the extract_clinical_disorders utility function
to extract and format clinical disorder information from pandas DataFrames.
"""

import pandas as pd
from perturbgen.src.utils import extract_clinical_disorders


def example_basic_extraction():
    """
    Example 1: Basic extraction of clinical disorders.
    
    This example shows how to extract all clinical disorders from a DataFrame
    and format them as a comma-separated string.
    """
    print("=" * 70)
    print("Example 1: Basic Extraction")
    print("=" * 70)
    
    # Create a sample DataFrame with clinical disorder information
    disease_df = pd.DataFrame({
        'Clinical disorder': ['Anemia', 'Leukemia', 'Thrombocytopenia', 'Neutropenia'],
        'Severity': ['Mild', 'Severe', 'Moderate', 'Mild'],
        'Patient_ID': ['P001', 'P002', 'P003', 'P004']
    })
    
    print("\nInput DataFrame:")
    print(disease_df)
    
    # Extract all clinical disorders
    disorders = extract_clinical_disorders(disease_df)
    
    print(f"\nExtracted disorders: {disorders}")
    print()


def example_filtered_extraction():
    """
    Example 2: Filtered extraction based on criteria.
    
    This example shows how to extract clinical disorders that match
    specific criteria (e.g., only severe cases).
    """
    print("=" * 70)
    print("Example 2: Filtered Extraction")
    print("=" * 70)
    
    # Create a sample DataFrame
    disease_df = pd.DataFrame({
        'Clinical disorder': ['Anemia', 'Leukemia', 'Thrombocytopenia', 'Neutropenia'],
        'Severity': ['Mild', 'Severe', 'Moderate', 'Severe'],
        'Treatment_Required': [False, True, False, True]
    })
    
    print("\nInput DataFrame:")
    print(disease_df)
    
    # Extract only severe disorders
    severe_disorders = extract_clinical_disorders(
        disease_df,
        filter_column='Severity',
        filter_value='Severe'
    )
    
    print(f"\nSevere disorders: {severe_disorders}")
    
    # Extract disorders requiring treatment
    treatment_disorders = extract_clinical_disorders(
        disease_df,
        filter_column='Treatment_Required',
        filter_value=True
    )
    
    print(f"Disorders requiring treatment: {treatment_disorders}")
    print()


def example_custom_column():
    """
    Example 3: Using custom column names.
    
    This example shows how to work with DataFrames that use different
    column names for disorder information.
    """
    print("=" * 70)
    print("Example 3: Custom Column Names")
    print("=" * 70)
    
    # Create a DataFrame with custom column names
    disease_df = pd.DataFrame({
        'Disease': ['Diabetes Type 2', 'Hypertension', 'Asthma'],
        'Status': ['Chronic', 'Chronic', 'Active'],
        'Age_Group': ['Adult', 'Senior', 'Child']
    })
    
    print("\nInput DataFrame:")
    print(disease_df)
    
    # Extract using custom column name
    diseases = extract_clinical_disorders(
        disease_df,
        column_name='Disease'
    )
    
    print(f"\nExtracted diseases: {diseases}")
    
    # Extract chronic conditions only
    chronic_diseases = extract_clinical_disorders(
        disease_df,
        column_name='Disease',
        filter_column='Status',
        filter_value='Chronic'
    )
    
    print(f"Chronic diseases: {chronic_diseases}")
    print()


def example_handling_duplicates():
    """
    Example 4: Handling duplicate disorders.
    
    This example shows how the function automatically handles duplicate
    entries and returns only unique disorders.
    """
    print("=" * 70)
    print("Example 4: Handling Duplicates")
    print("=" * 70)
    
    # Create a DataFrame with duplicate disorders
    disease_df = pd.DataFrame({
        'Clinical disorder': ['Anemia', 'Leukemia', 'Anemia', 'Leukemia', 'Anemia'],
        'Patient_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'Date': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05']
    })
    
    print("\nInput DataFrame (with duplicates):")
    print(disease_df)
    
    # Extract unique disorders
    unique_disorders = extract_clinical_disorders(disease_df)
    
    print(f"\nUnique disorders: {unique_disorders}")
    print("Note: Duplicates are automatically removed")
    print()


def example_real_world_use_case():
    """
    Example 5: Real-world use case - Patient cohort analysis.
    
    This example demonstrates a practical use case where you might want to
    extract clinical disorders for a specific patient cohort.
    """
    print("=" * 70)
    print("Example 5: Real-world Use Case - Patient Cohort Analysis")
    print("=" * 70)
    
    # Create a more realistic DataFrame
    disease_df = pd.DataFrame({
        'Clinical disorder': [
            'Acute Myeloid Leukemia',
            'Myelodysplastic Syndrome',
            'Acute Lymphoblastic Leukemia',
            'Chronic Myeloid Leukemia',
            'Aplastic Anemia',
            'Thrombocytopenia'
        ],
        'Disease_Category': [
            'Leukemia', 'MDS', 'Leukemia', 'Leukemia', 'Anemia', 'Platelet Disorder'
        ],
        'Study_Cohort': [
            'HSPC', 'HSPC', 'HSPC', 'Control', 'HSPC', 'Control'
        ],
        'Sample_Count': [25, 18, 12, 8, 15, 10]
    })
    
    print("\nInput DataFrame:")
    print(disease_df)
    
    # Extract disorders in the HSPC study cohort
    hspc_disorders = extract_clinical_disorders(
        disease_df,
        filter_column='Study_Cohort',
        filter_value='HSPC'
    )
    
    print(f"\nDisorders in HSPC cohort: {hspc_disorders}")
    
    # Extract only leukemia subtypes
    leukemia_types = extract_clinical_disorders(
        disease_df,
        filter_column='Disease_Category',
        filter_value='Leukemia'
    )
    
    print(f"Leukemia subtypes: {leukemia_types}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Clinical Disorder Extraction Examples")
    print("=" * 70)
    print()
    
    example_basic_extraction()
    example_filtered_extraction()
    example_custom_column()
    example_handling_duplicates()
    example_real_world_use_case()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
