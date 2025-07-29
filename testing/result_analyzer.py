# testing/result_analyzer.py
"""
Result Analysis Module for LLM-ATC-HAL Framework
Handles statistical analysis and performance metrics calculation
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime

from .test_executor import TestResult


class ResultAnalyzer:
    """Analyzes test results and generates statistical summaries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_results(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results and generate statistical summaries"""
        self.logger.info("Analyzing test results...")
        
        if not test_results:
            self.logger.warning("No test results to analyze")
            return {'error': 'No test results available'}
        
        # Convert results to DataFrame for analysis
        results_data = []
        for result in test_results:
            if not result.errors:  # Only analyze successful tests
                results_data.append({
                    'test_id': result.test_id,
                    'scenario_type': result.scenario_type,
                    'complexity_level': result.complexity_level,
                    'model_used': result.model_used,
                    'response_time': result.response_time,
                    'hallucination_detected': result.hallucination_detected,
                    'confidence_score': result.confidence_score,
                    'safety_margin': result.safety_margin,
                    'icao_compliant': result.icao_compliant,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall
                })
        
        if not results_data:
            return {'error': 'No successful test results to analyze'}
        
        df = pd.DataFrame(results_data)
        
        # Statistical analysis
        statistical_summary = self._calculate_statistical_summary(test_results, df)
        
        # Performance analysis by model
        model_analysis = self._analyze_by_model(df)
        
        # Complexity analysis
        complexity_analysis = self._analyze_by_complexity(df)
        
        # Hallucination analysis
        hallucination_analysis = self._analyze_hallucinations(df)
        
        # Safety analysis
        safety_analysis = self._analyze_safety_metrics(df)
        
        return {
            'statistical_summary': statistical_summary,
            'model_analysis': model_analysis,
            'complexity_analysis': complexity_analysis,
            'hallucination_analysis': hallucination_analysis,
            'safety_analysis': safety_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_statistical_summary(self, test_results: List[TestResult], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical summary"""
        return {
            'total_tests': len(test_results),
            'successful_tests': len(df),
            'error_rate': (len(test_results) - len(df)) / len(test_results) if test_results else 0,
            
            'response_time': {
                'mean': float(df['response_time'].mean()) if not df.empty else 0,
                'std': float(df['response_time'].std()) if not df.empty else 0,
                'min': float(df['response_time'].min()) if not df.empty else 0,
                'max': float(df['response_time'].max()) if not df.empty else 0,
                'p95': float(df['response_time'].quantile(0.95)) if not df.empty else 0
            },
            
            'confidence_score': {
                'mean': float(df['confidence_score'].mean()) if not df.empty else 0,
                'std': float(df['confidence_score'].std()) if not df.empty else 0,
                'min': float(df['confidence_score'].min()) if not df.empty else 0,
                'max': float(df['confidence_score'].max()) if not df.empty else 0
            },
            
            'safety_margin': {
                'mean': float(df['safety_margin'].mean()) if not df.empty else 0,
                'std': float(df['safety_margin'].std()) if not df.empty else 0,
                'min': float(df['safety_margin'].min()) if not df.empty else 0,
                'max': float(df['safety_margin'].max()) if not df.empty else 0
            },
            
            'performance_metrics': {
                'accuracy': {
                    'mean': float(df['accuracy'].mean()) if not df.empty else 0,
                    'std': float(df['accuracy'].std()) if not df.empty else 0
                },
                'precision': {
                    'mean': float(df['precision'].mean()) if not df.empty else 0,
                    'std': float(df['precision'].std()) if not df.empty else 0
                },
                'recall': {
                    'mean': float(df['recall'].mean()) if not df.empty else 0,
                    'std': float(df['recall'].std()) if not df.empty else 0
                }
            }
        }
    
    def _analyze_by_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by model"""
        if df.empty:
            return {}
        
        model_stats = {}
        
        for model in df['model_used'].unique():
            model_df = df[df['model_used'] == model]
            
            model_stats[model] = {
                'test_count': len(model_df),
                'avg_response_time': float(model_df['response_time'].mean()),
                'avg_confidence': float(model_df['confidence_score'].mean()),
                'avg_safety_margin': float(model_df['safety_margin'].mean()),
                'hallucination_rate': float(model_df['hallucination_detected'].mean()),
                'icao_compliance_rate': float(model_df['icao_compliant'].mean()),
                'avg_accuracy': float(model_df['accuracy'].mean()),
                'avg_precision': float(model_df['precision'].mean()),
                'avg_recall': float(model_df['recall'].mean())
            }
        
        return model_stats
    
    def _analyze_by_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by complexity level"""
        if df.empty:
            return {}
        
        complexity_stats = {}
        
        for complexity in df['complexity_level'].unique():
            complexity_df = df[df['complexity_level'] == complexity]
            
            complexity_stats[complexity] = {
                'test_count': len(complexity_df),
                'avg_response_time': float(complexity_df['response_time'].mean()),
                'avg_confidence': float(complexity_df['confidence_score'].mean()),
                'avg_safety_margin': float(complexity_df['safety_margin'].mean()),
                'hallucination_rate': float(complexity_df['hallucination_detected'].mean()),
                'icao_compliance_rate': float(complexity_df['icao_compliant'].mean()),
                'avg_accuracy': float(complexity_df['accuracy'].mean())
            }
        
        return complexity_stats
    
    def _analyze_hallucinations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hallucination detection results"""
        if df.empty:
            return {}
        
        total_tests = len(df)
        hallucinations_detected = df['hallucination_detected'].sum()
        
        return {
            'overall_hallucination_rate': float(hallucinations_detected / total_tests) if total_tests > 0 else 0,
            'hallucinations_by_model': {
                model: float(group['hallucination_detected'].mean())
                for model, group in df.groupby('model_used')
            },
            'hallucinations_by_complexity': {
                complexity: float(group['hallucination_detected'].mean())
                for complexity, group in df.groupby('complexity_level')
            },
            'confidence_when_hallucinating': float(
                df[df['hallucination_detected']]['confidence_score'].mean()
            ) if hallucinations_detected > 0 else 0,
            'confidence_when_not_hallucinating': float(
                df[~df['hallucination_detected']]['confidence_score'].mean()
            ) if (total_tests - hallucinations_detected) > 0 else 0
        }
    
    def _analyze_safety_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze safety-related metrics"""
        if df.empty:
            return {}
        
        total_tests = len(df)
        icao_compliant = df['icao_compliant'].sum()
        
        # Safety margin analysis
        low_safety_margin = (df['safety_margin'] < 0.3).sum()
        critical_safety_margin = (df['safety_margin'] < 0.1).sum()
        
        return {
            'icao_compliance_rate': float(icao_compliant / total_tests) if total_tests > 0 else 0,
            'low_safety_margin_rate': float(low_safety_margin / total_tests) if total_tests > 0 else 0,
            'critical_safety_margin_rate': float(critical_safety_margin / total_tests) if total_tests > 0 else 0,
            'safety_margin_stats': {
                'mean': float(df['safety_margin'].mean()),
                'std': float(df['safety_margin'].std()),
                'min': float(df['safety_margin'].min()),
                'p5': float(df['safety_margin'].quantile(0.05)),
                'p95': float(df['safety_margin'].quantile(0.95))
            },
            'safety_by_model': {
                model: {
                    'avg_safety_margin': float(group['safety_margin'].mean()),
                    'icao_compliance_rate': float(group['icao_compliant'].mean())
                }
                for model, group in df.groupby('model_used')
            }
        }
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """Generate visualization plots and save to output directory"""
        if df.empty:
            self.logger.warning("No data available for visualization")
            return []
        
        plot_files = []
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Response time distribution by model
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='model_used', y='response_time')
            plt.title('Response Time Distribution by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_file = f"{output_dir}/response_time_by_model.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            # Hallucination rate by complexity
            plt.figure(figsize=(10, 6))
            hallucination_by_complexity = df.groupby('complexity_level')['hallucination_detected'].mean()
            sns.barplot(x=hallucination_by_complexity.index, y=hallucination_by_complexity.values)
            plt.title('Hallucination Rate by Complexity Level')
            plt.ylabel('Hallucination Rate')
            plt.tight_layout()
            plot_file = f"{output_dir}/hallucination_by_complexity.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            # Safety margin distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df['safety_margin'], bins=30, alpha=0.7)
            plt.axvline(df['safety_margin'].mean(), color='red', linestyle='--', label='Mean')
            plt.axvline(0.3, color='orange', linestyle='--', label='Low Threshold')
            plt.axvline(0.1, color='red', linestyle='--', label='Critical Threshold')
            plt.title('Safety Margin Distribution')
            plt.xlabel('Safety Margin')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plot_file = f"{output_dir}/safety_margin_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            # Performance correlation matrix
            performance_cols = ['response_time', 'confidence_score', 'safety_margin', 'accuracy', 'precision', 'recall']
            available_cols = [col for col in performance_cols if col in df.columns]
            
            if len(available_cols) > 1:
                plt.figure(figsize=(8, 6))
                correlation_matrix = df[available_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Performance Metrics Correlation')
                plt.tight_layout()
                plot_file = f"{output_dir}/performance_correlation.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_file)
            
            self.logger.info(f"Generated {len(plot_files)} visualization plots")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        return plot_files
    
    def export_results_summary(self, analysis_results: Dict[str, Any], output_file: str):
        """Export analysis results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            self.logger.info(f"Analysis results exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to export analysis results: {e}")
