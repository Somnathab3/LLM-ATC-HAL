#!/usr/bin/env python3
"""
Model Comparison Analysis
Analyzes the performance comparison between fine-tuned and baseline models
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def analyze_model_performance(baseline_csv_path, fine_tuned_csv_path):
    """Analyze and compare model performance from CSV files"""
    
    print("🔍 Loading Model Performance Data...")
    
    # Load CSV files
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        print(f"✅ Loaded baseline model data: {len(baseline_df)} scenarios")
    except Exception as e:
        print(f"❌ Error loading baseline data: {e}")
        return
    
    try:
        fine_tuned_df = pd.read_csv(fine_tuned_csv_path)
        print(f"✅ Loaded fine-tuned model data: {len(fine_tuned_df)} scenarios")
    except Exception as e:
        print(f"❌ Error loading fine-tuned data: {e}")
        return
    
    print("\n" + "="*80)
    print("📊 MODEL PERFORMANCE COMPARISON ANALYSIS")
    print("="*80)
    
    # Scenario counts
    print(f"\n📝 SCENARIO COUNTS:")
    print(f"   Baseline Model:    {len(baseline_df):,} scenarios")
    print(f"   Fine-tuned Model:  {len(fine_tuned_df):,} scenarios")
    
    # Success rates (based on resolution_success column)
    baseline_success = baseline_df['resolution_success'].sum() if 'resolution_success' in baseline_df.columns else 0
    fine_tuned_success = fine_tuned_df['resolution_success'].sum() if 'resolution_success' in fine_tuned_df.columns else 0
    
    baseline_success_rate = (baseline_success / len(baseline_df)) * 100 if len(baseline_df) > 0 else 0
    fine_tuned_success_rate = (fine_tuned_success / len(fine_tuned_df)) * 100 if len(fine_tuned_df) > 0 else 0
    
    print(f"\n🎯 SUCCESS RATES:")
    print(f"   Baseline Model:    {baseline_success_rate:.2f}% ({baseline_success}/{len(baseline_df)})")
    print(f"   Fine-tuned Model:  {fine_tuned_success_rate:.2f}% ({fine_tuned_success}/{len(fine_tuned_df)})")
    print(f"   Improvement:       {fine_tuned_success_rate - baseline_success_rate:+.2f}%")
    
    # Detection accuracy
    baseline_detection_acc = baseline_df['detection_accuracy'].mean() if 'detection_accuracy' in baseline_df.columns else 0
    fine_tuned_detection_acc = fine_tuned_df['detection_accuracy'].mean() if 'detection_accuracy' in fine_tuned_df.columns else 0
    
    print(f"\n🎯 DETECTION ACCURACY:")
    print(f"   Baseline Model:    {baseline_detection_acc:.4f}")
    print(f"   Fine-tuned Model:  {fine_tuned_detection_acc:.4f}")
    print(f"   Improvement:       {fine_tuned_detection_acc - baseline_detection_acc:+.4f}")
    
    # Response times
    baseline_time = baseline_df['execution_time_ms'].mean() if 'execution_time_ms' in baseline_df.columns else 0
    fine_tuned_time = fine_tuned_df['execution_time_ms'].mean() if 'execution_time_ms' in fine_tuned_df.columns else 0
    
    print(f"\n⏱️  RESPONSE TIMES:")
    print(f"   Baseline Model:    {baseline_time:.1f}ms")
    print(f"   Fine-tuned Model:  {fine_tuned_time:.1f}ms")
    print(f"   Speed Improvement: {baseline_time - fine_tuned_time:+.1f}ms")
    
    # LLM Confidence
    baseline_conf = baseline_df['llm_confidence'].mean() if 'llm_confidence' in baseline_df.columns else 0
    fine_tuned_conf = fine_tuned_df['llm_confidence'].mean() if 'llm_confidence' in fine_tuned_df.columns else 0
    
    print(f"\n🤖 LLM CONFIDENCE:")
    print(f"   Baseline Model:    {baseline_conf:.3f}")
    print(f"   Fine-tuned Model:  {fine_tuned_conf:.3f}")
    print(f"   Improvement:       {fine_tuned_conf - baseline_conf:+.3f}")
    
    # Scenario type breakdown
    print(f"\n📋 SCENARIO TYPE BREAKDOWN:")
    
    if 'scenario_type' in baseline_df.columns:
        baseline_types = baseline_df['scenario_type'].value_counts()
        print(f"   Baseline Model:")
        for scenario_type, count in baseline_types.items():
            print(f"     {scenario_type.title()}: {count} scenarios")
    
    if 'scenario_type' in fine_tuned_df.columns:
        fine_tuned_types = fine_tuned_df['scenario_type'].value_counts()
        print(f"   Fine-tuned Model:")
        for scenario_type, count in fine_tuned_types.items():
            print(f"     {scenario_type.title()}: {count} scenarios")
    
    # Ground truth vs LLM conflicts analysis
    print(f"\n🎯 CONFLICT DETECTION ANALYSIS:")
    
    # Calculate confusion matrix components
    def analyze_conflicts(df, model_name):
        if 'ground_truth_conflicts' not in df.columns or 'llm_conflicts' not in df.columns:
            return
            
        gt_conflicts = df['ground_truth_conflicts']
        llm_conflicts = df['llm_conflicts']
        
        # True positives: ground truth has conflict AND LLM detected conflict
        tp = ((gt_conflicts > 0) & (llm_conflicts > 0)).sum()
        
        # True negatives: ground truth has no conflict AND LLM detected no conflict
        tn = ((gt_conflicts == 0) & (llm_conflicts == 0)).sum()
        
        # False positives: ground truth has no conflict BUT LLM detected conflict
        fp = ((gt_conflicts == 0) & (llm_conflicts > 0)).sum()
        
        # False negatives: ground truth has conflict BUT LLM detected no conflict
        fn = ((gt_conflicts > 0) & (llm_conflicts == 0)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        
        print(f"   {model_name} Confusion Matrix:")
        print(f"     True Positives:   {tp}")
        print(f"     True Negatives:   {tn}")
        print(f"     False Positives:  {fp}")
        print(f"     False Negatives:  {fn}")
        print(f"   {model_name} Metrics:")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1-Score:  {f1_score:.4f}")
        print(f"     Accuracy:  {accuracy:.4f}")
        
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy}
    
    baseline_metrics = analyze_conflicts(baseline_df, "Baseline Model")
    print()
    fine_tuned_metrics = analyze_conflicts(fine_tuned_df, "Fine-tuned Model")
    
    # Summary and conclusions
    print(f"\n" + "="*80)
    print("🎯 SUMMARY & CONCLUSIONS")
    print("="*80)
    
    improvements = []
    concerns = []
    
    # Success rate analysis
    success_improvement = fine_tuned_success_rate - baseline_success_rate
    if success_improvement > 1.0:
        improvements.append(f"Success rate improved by {success_improvement:.2f}%")
    elif success_improvement < -1.0:
        concerns.append(f"Success rate decreased by {abs(success_improvement):.2f}%")
    
    # Response time analysis
    time_improvement = baseline_time - fine_tuned_time
    if time_improvement > 100:
        improvements.append(f"Response time improved by {time_improvement:.1f}ms")
    elif time_improvement < -100:
        concerns.append(f"Response time increased by {abs(time_improvement):.1f}ms")
    
    # Detection analysis
    if baseline_metrics and fine_tuned_metrics:
        f1_improvement = fine_tuned_metrics['f1_score'] - baseline_metrics['f1_score']
        if f1_improvement > 0.01:
            improvements.append(f"F1-score improved by {f1_improvement:.4f}")
        elif f1_improvement < -0.01:
            concerns.append(f"F1-score decreased by {abs(f1_improvement):.4f}")
    
    print("✅ KEY IMPROVEMENTS:")
    if improvements:
        for improvement in improvements:
            print(f"   • {improvement}")
    else:
        print("   • No significant improvements detected")
    
    print("\n⚠️  AREAS FOR CONCERN:")
    if concerns:
        for concern in concerns:
            print(f"   • {concern}")
    else:
        print("   • No significant concerns detected")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if len(improvements) > len(concerns):
        print("   ✅ Fine-tuned model shows measurable improvements")
    elif len(concerns) > len(improvements):
        print("   ⚠️  Fine-tuned model may need further optimization")
    else:
        print("   📊 Models show similar performance levels")
    
    print(f"\n📊 MODEL RECOMMENDATION:")
    if fine_tuned_success_rate > baseline_success_rate and fine_tuned_time < baseline_time:
        print("   🌟 RECOMMENDED: Use fine-tuned model for better performance")
    elif fine_tuned_success_rate > baseline_success_rate:
        print("   ✅ RECOMMENDED: Use fine-tuned model for better accuracy")
    elif fine_tuned_time < baseline_time:
        print("   ⚡ RECOMMENDED: Use fine-tuned model for faster response")
    else:
        print("   📋 NEUTRAL: Both models have similar performance characteristics")


if __name__ == "__main__":
    # Use the actual file paths from the model comparison test
    baseline_path = "experiments/model_comparison_test/baseline_model_test/benchmark_645f4063_20250803_022729/detection_comparison.csv"
    fine_tuned_path = "experiments/model_comparison_test/fine-tuned_model_test/benchmark_acbe1913_20250803_030838/detection_comparison.csv"
    
    analyze_model_performance(baseline_path, fine_tuned_path)
