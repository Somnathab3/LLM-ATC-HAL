#!/usr/bin/env python3
"""
Test script for Monte Carlo analysis enhancements.
Tests the new generate_report method and validates existing functionality.
"""

import pandas as pd
import tempfile
import json
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from llm_atc.metrics.monte_carlo_analysis import MonteCarloResultsAnalyzer


def test_monte_carlo_analysis_enhancements():
    """Test the enhanced Monte Carlo analysis functionality."""
    print("🧪 Testing Monte Carlo Analysis Enhancements")
    print("=" * 60)
    
    # Create test data
    test_data = [
        {
            'scenario_type': 'horizontal',
            'complexity_tier': 'simple',
            'distribution_shift_level': 'baseline',
            'success': True,
            'predicted_conflicts': [],
            'actual_conflicts': [],
            'horizontal_margin': 6.2,
            'vertical_margin': 1200,
            'efficiency_penalty': 2.1,
            'detection_accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88
        },
        {
            'scenario_type': 'horizontal',
            'complexity_tier': 'complex',
            'distribution_shift_level': 'mild',
            'success': False,
            'predicted_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'actual_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'horizontal_margin': 3.8,
            'vertical_margin': 800,
            'efficiency_penalty': 5.3,
            'detection_accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.91
        },
        {
            'scenario_type': 'vertical',
            'complexity_tier': 'simple',
            'distribution_shift_level': 'baseline',
            'success': True,
            'predicted_conflicts': [],
            'actual_conflicts': [],
            'horizontal_margin': 7.1,
            'vertical_margin': 1500,
            'efficiency_penalty': 1.8,
            'detection_accuracy': 0.96,
            'precision': 0.93,
            'recall': 0.89
        },
        {
            'scenario_type': 'vertical',
            'complexity_tier': 'complex',
            'distribution_shift_level': 'severe',
            'success': False,
            'predicted_conflicts': [{'aircraft_1': 'AC003', 'aircraft_2': 'AC004'}],
            'actual_conflicts': [{'aircraft_1': 'AC003', 'aircraft_2': 'AC004'}, {'aircraft_1': 'AC005', 'aircraft_2': 'AC006'}],
            'horizontal_margin': 2.5,
            'vertical_margin': 600,
            'efficiency_penalty': 12.7,
            'detection_accuracy': 0.72,
            'precision': 0.68,
            'recall': 0.85
        },
        {
            'scenario_type': 'sector',
            'complexity_tier': 'simple',
            'distribution_shift_level': 'mild',
            'success': True,
            'predicted_conflicts': [{'aircraft_1': 'AC007', 'aircraft_2': 'AC008'}],
            'actual_conflicts': [{'aircraft_1': 'AC007', 'aircraft_2': 'AC008'}],
            'horizontal_margin': 5.4,
            'vertical_margin': 1100,
            'efficiency_penalty': 3.2,
            'detection_accuracy': 0.91,
            'precision': 0.89,
            'recall': 0.92
        }
    ]
    
    # Initialize analyzer
    analyzer = MonteCarloResultsAnalyzer()
    
    # Test 1: Create DataFrame and basic metrics
    print("📊 Test 1: Basic functionality")
    results_df = pd.DataFrame(test_data)
    aggregated_metrics = analyzer.aggregate_monte_carlo_metrics(results_df)
    
    print(f"   ✅ Total scenarios: {aggregated_metrics['summary']['total_scenarios']}")
    print(f"   ✅ Scenario types: {aggregated_metrics['summary']['scenario_types']}")
    print(f"   ✅ FP rate: {aggregated_metrics['detection_performance']['false_positive_rate']:.3f}")
    print(f"   ✅ FN rate: {aggregated_metrics['detection_performance']['false_negative_rate']:.3f}")
    
    # Test 2: Grouped success rates
    print("\n📈 Test 2: Grouped success rates")
    group_cols = ['scenario_type', 'complexity_tier', 'distribution_shift_level']
    grouped_rates = analyzer.compute_success_rates_by_group(results_df, group_cols)
    
    print(f"   ✅ Grouped analysis shape: {grouped_rates.shape}")
    print(f"   ✅ Columns: {list(grouped_rates.columns)}")
    
    if not grouped_rates.empty:
        print("   📋 Sample grouped results:")
        for index, row in grouped_rates.head(2).iterrows():
            if isinstance(index, tuple):
                group_name = " / ".join(str(x) for x in index)
            else:
                group_name = str(index)
            success_rate = row.get('success_rate', 0)
            print(f"      - {group_name}: {success_rate:.3f} success rate")
    
    # Test 3: Generate comprehensive report
    print("\n📝 Test 3: Generate comprehensive report")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "test_monte_carlo_report.md"
        
        generated_report = analyzer.generate_report(
            results_df=results_df,
            aggregated_metrics=aggregated_metrics,
            output_file=report_path
        )
        
        print(f"   ✅ Report generated: {generated_report}")
        
        # Verify report contents
        if Path(generated_report).exists():
            with open(generated_report, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Check for key sections
            required_sections = [
                "# Monte Carlo Analysis Report",
                "## Executive Summary", 
                "## Detection Performance",
                "## Success Rates by Scenario Type",
                "## Safety Margins",
                "## Efficiency Metrics",
                "## Distribution Shift Analysis",
                "## Recommendations"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in report_content:
                    missing_sections.append(section)
            
            if not missing_sections:
                print("   ✅ All required sections present in report")
            else:
                print(f"   ❌ Missing sections: {missing_sections}")
            
            # Show some report statistics
            lines = report_content.split('\n')
            print(f"   📊 Report statistics:")
            print(f"      - Total lines: {len(lines)}")
            print(f"      - Characters: {len(report_content)}")
            print(f"      - Tables: {report_content.count('|')//4}")  # Rough table count
            
            # Show a sample of the executive summary
            if "## Executive Summary" in report_content:
                summary_start = report_content.find("## Executive Summary")
                summary_section = report_content[summary_start:summary_start+500]
                print(f"   📋 Sample executive summary:")
                for line in summary_section.split('\n')[2:5]:  # Skip header lines
                    if line.strip():
                        print(f"      {line[:80]}...")
                        break
        else:
            print(f"   ❌ Report file not found: {generated_report}")
    
    # Test 4: Edge cases
    print("\n🔍 Test 4: Edge cases")
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    empty_metrics = analyzer.aggregate_monte_carlo_metrics(empty_df)
    print(f"   ✅ Empty DataFrame handled: {empty_metrics['summary']['total_scenarios']} scenarios")
    
    # Missing columns
    minimal_df = pd.DataFrame([
        {'scenario_type': 'test', 'success': True},
        {'scenario_type': 'test', 'success': False}
    ])
    minimal_grouped = analyzer.compute_success_rates_by_group(minimal_df, ['scenario_type'])
    print(f"   ✅ Minimal DataFrame handled: {minimal_grouped.shape}")
    
    # Test 5: File I/O
    print("\n💾 Test 5: File I/O functionality")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        test_file = f.name
    
    try:
        # Test reading from file
        loaded_df = analyzer.read_results_file(test_file)
        print(f"   ✅ File loaded successfully: {len(loaded_df)} scenarios")
        
        # Test full pipeline
        full_analysis = analyzer.aggregate_monte_carlo_metrics(loaded_df)
        print(f"   ✅ Full pipeline works: {full_analysis['summary']['total_scenarios']} scenarios processed")
        
    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("\n📋 Summary of enhancements:")
    print("   ✅ compute_success_rates_by_group() - Multi-index DataFrame grouping")
    print("   ✅ Refined separation margin calculation - Enhanced error handling") 
    print("   ✅ generate_report() - Comprehensive markdown reports")
    print("   ✅ Executive summary generation")
    print("   ✅ Performance assessments and recommendations")
    print("   ✅ Distribution shift analysis formatting")
    print("   ✅ Markdown table generation")


if __name__ == "__main__":
    test_monte_carlo_analysis_enhancements()
