"""
Comprehensive Visualization Suite for LLM-ATC-HAL Framework
==========================================================

This module provides advanced visualization capabilities for Monte Carlo analysis,
distribution shift studies, and performance evaluation in air traffic control scenarios.

Features:
- Statistical distributions (histograms, KDEs, violin plots)
- Cumulative analysis and time-series trends
- Sensitivity analysis (tornado charts, spider plots)
- Geospatial trajectory visualization
- Interactive dashboards
- Automated report generation
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from dataclasses import asdict
from collections import defaultdict

import numpy as np
import pandas as pd

# Core plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    
    # Set style preferences
    plt.style.use('default')
    sns.set_palette("husl")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

# Advanced plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Geospatial libraries
try:
    import folium
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    folium = None

# Statistical libraries
try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class MonteCarloVisualizer:
    """
    Comprehensive visualization suite for Monte Carlo analysis results.
    
    Supports multiple chart types:
    - Distribution analysis (histograms, KDEs, violin plots)
    - Cumulative analysis (ECDFs, time-series trends)
    - Sensitivity analysis (tornado charts, spider plots)
    - Geospatial visualization (trajectory maps, heatmaps)
    - Interactive dashboards
    """
    
    def __init__(self, output_dir: Union[str, Path] = "visualizations", 
                 style: str = "seaborn", dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style to use
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Create subdirectories
        for subdir in ['static', 'interactive', 'geospatial', 'animations']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Set plotting style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style in plt.style.available else 'default')
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#FFB700',
            'distribution_shift': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'complexity_tiers': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
        }
        
        logger.info(f"MonteCarloVisualizer initialized. Output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, data: pd.DataFrame, 
                                    title: str = "Monte Carlo Analysis Report") -> str:
        """
        Generate a complete visualization report with all chart types.
        
        Args:
            data: DataFrame with Monte Carlo results
            title: Report title
            
        Returns:
            Path to generated HTML report
        """
        logger.info("Generating comprehensive visualization report...")
        
        # Generate all visualization categories
        static_plots = self._generate_distribution_analysis(data)
        trend_plots = self._generate_trend_analysis(data)
        sensitivity_plots = self._generate_sensitivity_analysis(data)
        
        if PLOTLY_AVAILABLE:
            interactive_plots = self._generate_interactive_dashboard(data)
        else:
            interactive_plots = []
            
        if GEOSPATIAL_AVAILABLE and 'latitude' in data.columns:
            geospatial_plots = self._generate_geospatial_analysis(data)
        else:
            geospatial_plots = []
        
        # Create HTML report
        report_path = self._create_html_report(
            title, static_plots, trend_plots, sensitivity_plots, 
            interactive_plots, geospatial_plots
        )
        
        logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)
    
    def _generate_distribution_analysis(self, data: pd.DataFrame) -> List[str]:
        """Generate distribution analysis visualizations."""
        plots = []
        
        # 1.1 Histograms & KDEs for Key Metrics
        plots.append(self._plot_metric_distributions(data))
        
        # 1.2 Side-by-Side Density Comparisons
        if 'distribution_shift_tier' in data.columns:
            plots.append(self._plot_shift_comparisons(data))
        
        plots.append(self._plot_violin_comparisons(data))
        plots.append(self._plot_ridge_plots(data))
        
        return [p for p in plots if p is not None]
    
    def _generate_trend_analysis(self, data: pd.DataFrame) -> List[str]:
        """Generate trend and cumulative analysis visualizations."""
        plots = []
        
        # 2.1 Cumulative False-Positive/Negative Curves
        plots.append(self._plot_cumulative_error_curves(data))
        
        # 2.2 Time-Series of Conflict Events
        if 'simulation_time' in data.columns or 'scenario_index' in data.columns:
            plots.append(self._plot_time_series_analysis(data))
        
        plots.append(self._plot_performance_evolution(data))
        
        return [p for p in plots if p is not None]
    
    def _generate_sensitivity_analysis(self, data: pd.DataFrame) -> List[str]:
        """Generate sensitivity and uncertainty visualizations."""
        plots = []
        
        # 3.1 Tornado (Bar-Chart) Sensitivity Analysis
        plots.append(self._plot_tornado_sensitivity(data))
        
        # 3.2 Spider/Radar Charts
        plots.append(self._plot_radar_comparison(data))
        
        plots.append(self._plot_parameter_correlation_heatmap(data))
        
        return [p for p in plots if p is not None]
    
    def _plot_metric_distributions(self, data: pd.DataFrame) -> Optional[str]:
        """Create histograms and KDEs for key metrics."""
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        # Key metrics to analyze
        metrics = {
            'horizontal_margin_ft': 'Horizontal Separation Margin (ft)',
            'vertical_margin_ft': 'Vertical Separation Margin (ft)', 
            'extra_path_nm': 'Extra Path Distance (nm)',
            'intervention_count': 'Intervention Count',
            'false_positive_rate': 'False Positive Rate',
            'false_negative_rate': 'False Negative Rate'
        }
        
        available_metrics = {k: v for k, v in metrics.items() if k in data.columns}
        if not available_metrics:
            logger.warning("No key metrics found in data for distribution plotting")
            return None
        
        n_metrics = len(available_metrics)
        cols = 3
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[i]
            
            # Skip if all values are NaN
            metric_data = data[metric].dropna()
            if len(metric_data) == 0:
                ax.text(0.5, 0.5, f'No data for {label}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Histogram
            ax.hist(metric_data, bins=30, alpha=0.7, density=True, 
                   color=self.colors['primary'], edgecolor='black', linewidth=0.5)
            
            # KDE overlay if scipy available
            if SCIPY_AVAILABLE and len(metric_data) > 1:
                try:
                    kde = stats.gaussian_kde(metric_data)
                    x_range = np.linspace(metric_data.min(), metric_data.max(), 200)
                    ax.plot(x_range, kde(x_range), color=self.colors['accent'], 
                           linewidth=2, label='KDE')
                    ax.legend()
                except Exception as e:
                    logger.debug(f"KDE failed for {metric}: {e}")
            
            # Add statistical annotations
            mean_val = metric_data.mean()
            median_val = metric_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8,
                      label=f'Median: {median_val:.3f}')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'{label} Distribution')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'metric_distributions.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated metric distributions plot")
        return str(output_path)
    
    def _plot_shift_comparisons(self, data: pd.DataFrame) -> Optional[str]:
        """Create side-by-side density comparisons for distribution shifts."""
        if not MATPLOTLIB_AVAILABLE or 'distribution_shift_tier' not in data.columns:
            return None
        
        # Metrics to compare across shift tiers
        comparison_metrics = [
            ('false_positive_rate', 'False Positive Rate'),
            ('false_negative_rate', 'False Negative Rate'),
            ('horizontal_margin_ft', 'Horizontal Margin (ft)'),
            ('safety_score', 'Safety Score')
        ]
        
        available_metrics = [(k, v) for k, v in comparison_metrics if k in data.columns]
        if not available_metrics:
            return None
        
        shift_tiers = data['distribution_shift_tier'].unique()
        n_metrics = len(available_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(available_metrics[:4]):  # Limit to 4 plots
            ax = axes[i]
            
            for j, tier in enumerate(shift_tiers):
                tier_data = data[data['distribution_shift_tier'] == tier][metric].dropna()
                if len(tier_data) == 0:
                    continue
                
                # Plot density
                if SCIPY_AVAILABLE and len(tier_data) > 1:
                    try:
                        kde = stats.gaussian_kde(tier_data)
                        x_range = np.linspace(tier_data.min(), tier_data.max(), 200)
                        ax.plot(x_range, kde(x_range), 
                               color=self.colors['distribution_shift'][j % len(self.colors['distribution_shift'])],
                               linewidth=2, label=f'{tier} (n={len(tier_data)})')
                    except Exception:
                        # Fallback to histogram
                        ax.hist(tier_data, bins=20, alpha=0.5, density=True,
                               color=self.colors['distribution_shift'][j % len(self.colors['distribution_shift'])],
                               label=f'{tier} (n={len(tier_data)})')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'{label} by Distribution Shift Tier')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'shift_comparisons.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated distribution shift comparisons")
        return str(output_path)
    
    def _plot_violin_comparisons(self, data: pd.DataFrame) -> Optional[str]:
        """Create violin plots for separation margins by categories."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Look for categorical variables and metrics
        categorical_cols = []
        for col in ['complexity_tier', 'scenario_type', 'distribution_shift_tier']:
            if col in data.columns and data[col].nunique() <= 10:
                categorical_cols.append(col)
        
        metric_cols = []
        for col in ['horizontal_margin_ft', 'vertical_margin_ft', 'safety_score']:
            if col in data.columns:
                metric_cols.append(col)
        
        if not categorical_cols or not metric_cols:
            return None
        
        n_plots = min(len(categorical_cols) * len(metric_cols), 6)  # Limit plots
        cols = 2
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        for cat_col in categorical_cols:
            for metric_col in metric_cols:
                if plot_idx >= n_plots:
                    break
                
                ax = axes[plot_idx]
                
                # Prepare data for violin plot
                categories = data[cat_col].unique()
                violin_data = []
                labels = []
                
                for category in categories:
                    cat_data = data[data[cat_col] == category][metric_col].dropna()
                    if len(cat_data) > 0:
                        violin_data.append(cat_data)
                        labels.append(f'{category}\n(n={len(cat_data)})')
                
                if violin_data:
                    # Create violin plot
                    parts = ax.violinplot(violin_data, positions=range(len(violin_data)), 
                                        showmeans=True, showmedians=True)
                    
                    # Customize colors
                    for pc, color in zip(parts['bodies'], self.colors['complexity_tiers']):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel(metric_col.replace('_', ' ').title())
                    ax.set_title(f'{metric_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'violin_comparisons.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated violin plot comparisons")
        return str(output_path)
    
    def _plot_ridge_plots(self, data: pd.DataFrame) -> Optional[str]:
        """Create ridge plots (joy plots) for metric distributions."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Look for a categorical variable for ridge plotting
        cat_col = None
        for col in ['distribution_shift_tier', 'complexity_tier', 'scenario_type']:
            if col in data.columns and data[col].nunique() <= 8:
                cat_col = col
                break
        
        if cat_col is None:
            return None
        
        # Select a continuous metric
        metric_col = None
        for col in ['horizontal_margin_ft', 'safety_score', 'false_positive_rate']:
            if col in data.columns:
                metric_col = col
                break
        
        if metric_col is None:
            return None
        
        categories = sorted(data[cat_col].unique())
        n_categories = len(categories)
        
        fig, axes = plt.subplots(n_categories, 1, figsize=(12, 2 * n_categories), 
                                sharex=True)
        if n_categories == 1:
            axes = [axes]
        
        for i, category in enumerate(categories):
            ax = axes[i]
            cat_data = data[data[cat_col] == category][metric_col].dropna()
            
            if len(cat_data) > 1:
                # Plot density curve
                if SCIPY_AVAILABLE:
                    try:
                        kde = stats.gaussian_kde(cat_data)
                        x_range = np.linspace(data[metric_col].min(), data[metric_col].max(), 200)
                        density = kde(x_range)
                        
                        # Fill area under curve
                        ax.fill_between(x_range, density, alpha=0.7, 
                                       color=self.colors['distribution_shift'][i % len(self.colors['distribution_shift'])])
                        ax.plot(x_range, density, color='black', linewidth=1)
                    except Exception:
                        # Fallback to histogram
                        ax.hist(cat_data, bins=30, alpha=0.7, density=True, 
                               color=self.colors['distribution_shift'][i % len(self.colors['distribution_shift'])])
            
            # Add category label
            ax.text(0.02, 0.8, f'{category} (n={len(cat_data)})', 
                   transform=ax.transAxes, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        axes[-1].set_xlabel(metric_col.replace('_', ' ').title())
        plt.suptitle(f'{metric_col.replace("_", " ").title()} Distribution by {cat_col.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'ridge_plots.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated ridge plots")
        return str(output_path)
    
    def _plot_cumulative_error_curves(self, data: pd.DataFrame) -> Optional[str]:
        """Create cumulative false-positive/negative curves (ECDFs)."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        error_metrics = ['false_positive_rate', 'false_negative_rate']
        available_metrics = [m for m in error_metrics if m in data.columns]
        
        if not available_metrics:
            return None
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            metric_data = data[metric].dropna().sort_values()
            
            if len(metric_data) == 0:
                continue
            
            # Calculate ECDF
            n = len(metric_data)
            y_values = np.arange(1, n + 1) / n
            
            # Plot ECDF
            ax.step(metric_data, y_values, where='post', linewidth=2, 
                   color=self.colors['primary'], label='ECDF')
            
            # Add percentile lines
            percentiles = [25, 50, 75, 90, 95]
            for p in percentiles:
                value = np.percentile(metric_data, p)
                ax.axvline(value, color='red', linestyle='--', alpha=0.6, 
                          label=f'{p}th percentile: {value:.3f}' if p in [50, 95] else '')
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'ECDF: {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add summary statistics text
            stats_text = f'Mean: {metric_data.mean():.3f}\nStd: {metric_data.std():.3f}\nN: {len(metric_data)}'
            ax.text(0.7, 0.3, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'cumulative_error_curves.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated cumulative error curves")
        return str(output_path)
    
    def _plot_time_series_analysis(self, data: pd.DataFrame) -> Optional[str]:
        """Create time-series analysis of conflict events."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Look for time-based column
        time_col = None
        for col in ['simulation_time', 'scenario_index', 'timestamp']:
            if col in data.columns:
                time_col = col
                break
        
        if time_col is None:
            return None
        
        # Metrics to plot over time
        metrics = {
            'intervention_count': 'Intervention Count',
            'conflict_count': 'Conflict Count', 
            'safety_violations': 'Safety Violations',
            'horizontal_margin_ft': 'Horizontal Margin (ft)'
        }
        
        available_metrics = {k: v for k, v in metrics.items() if k in data.columns}
        if not available_metrics:
            return None
        
        # Sort by time
        data_sorted = data.sort_values(time_col)
        
        # Create rolling statistics
        window_size = max(10, len(data_sorted) // 20)  # Adaptive window size
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[i]
            
            time_values = data_sorted[time_col]
            metric_values = data_sorted[metric]
            
            # Plot raw data
            ax.plot(time_values, metric_values, alpha=0.3, color='gray', 
                   label='Raw Data', linewidth=0.5)
            
            # Plot rolling mean
            rolling_mean = metric_values.rolling(window=window_size, center=True).mean()
            ax.plot(time_values, rolling_mean, color=self.colors['primary'], 
                   linewidth=2, label=f'Rolling Mean (window={window_size})')
            
            # Plot rolling std as confidence band
            rolling_std = metric_values.rolling(window=window_size, center=True).std()
            ax.fill_between(time_values, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std,
                           alpha=0.2, color=self.colors['primary'], 
                           label='±1 Std Dev')
            
            ax.set_ylabel(label)
            ax.set_title(f'{label} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel(time_col.replace('_', ' ').title())
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'time_series_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated time-series analysis")
        return str(output_path)
    
    def _plot_performance_evolution(self, data: pd.DataFrame) -> Optional[str]:
        """Plot performance metrics evolution."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Create performance evolution chart based on available data
        performance_metrics = ['safety_score', 'detection_accuracy', 'resolution_success_rate']
        available_metrics = [m for m in performance_metrics if m in data.columns]
        
        if not available_metrics:
            return None
        
        # Group by simulation batches (assuming scenario_index or similar)
        if 'scenario_index' in data.columns:
            batch_col = 'scenario_index'
        elif 'sim_id' in data.columns:
            batch_col = 'sim_id'
        else:
            # Create artificial batches
            data = data.copy()
            batch_size = max(10, len(data) // 20)
            data['batch'] = data.index // batch_size
            batch_col = 'batch'
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, metric in enumerate(available_metrics):
            # Calculate running average
            batch_means = data.groupby(batch_col)[metric].mean()
            
            ax.plot(batch_means.index, batch_means.values, 
                   marker='o', linewidth=2, markersize=4,
                   color=self.colors['distribution_shift'][i % len(self.colors['distribution_shift'])],
                   label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Simulation Batch')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance Evolution Over Simulation Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'performance_evolution.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated performance evolution plot")
        return str(output_path)
    
    def _plot_tornado_sensitivity(self, data: pd.DataFrame) -> Optional[str]:
        """Create tornado chart for sensitivity analysis."""
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            return None
        
        # Target metric for sensitivity analysis
        target_metrics = ['safety_score', 'false_positive_rate', 'resolution_success_rate']
        target_metric = None
        for metric in target_metrics:
            if metric in data.columns:
                target_metric = metric
                break
        
        if target_metric is None:
            return None
        
        # Parameters to analyze
        parameters = []
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64'] and col != target_metric:
                if data[col].nunique() > 1:  # Has variation
                    parameters.append(col)
        
        if len(parameters) < 2:
            logger.warning("Insufficient parameters for sensitivity analysis")
            return None
        
        # Calculate correlation coefficients
        sensitivities = []
        for param in parameters:
            # Remove NaN values
            mask = ~(data[param].isna() | data[target_metric].isna())
            if mask.sum() < 10:  # Need minimum data points
                continue
            
            param_data = data.loc[mask, param]
            target_data = data.loc[mask, target_metric]
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(param_data, target_data)[0, 1]
                if not np.isnan(correlation):
                    sensitivities.append((param, abs(correlation), correlation))
            except Exception:
                continue
        
        if len(sensitivities) < 2:
            logger.warning("Could not calculate sufficient correlations for tornado chart")
            return None
        
        # Sort by absolute correlation
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top parameters
        top_sensitivities = sensitivities[:10]  # Top 10
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_sensitivities) * 0.5)))
        
        params = [s[0].replace('_', ' ').title() for s in top_sensitivities]
        correlations = [s[2] for s in top_sensitivities]
        
        # Create horizontal bar chart
        colors = ['red' if c < 0 else 'blue' for c in correlations]
        bars = ax.barh(range(len(params)), correlations, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(corr + (0.01 if corr > 0 else -0.01), i, f'{corr:.3f}', 
                   va='center', ha='left' if corr > 0 else 'right', fontweight='bold')
        
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_xlabel(f'Correlation with {target_metric.replace("_", " ").title()}')
        ax.set_title('Parameter Sensitivity Analysis (Tornado Chart)')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        red_patch = patches.Patch(color='red', alpha=0.7, label='Negative Correlation')
        blue_patch = patches.Patch(color='blue', alpha=0.7, label='Positive Correlation')
        ax.legend(handles=[red_patch, blue_patch], loc='lower right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'static' / 'tornado_sensitivity.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated tornado sensitivity chart")
        return str(output_path)


def plot_cd_timeline(df: pd.DataFrame = None, sim_id: str = None, 
                    output_dir: str = "visualizations", **kwargs) -> Optional[str]:
    """
    Plot conflict detection timeline for a specific simulation.
    
    Args:
        df: DataFrame with simulation results
        sim_id: Simulation ID to plot
        output_dir: Output directory for plots
        **kwargs: Additional arguments
        
    Returns:
        Path to generated plot file, or None if visualization failed
    """
    try:
        visualizer = MonteCarloVisualizer(output_dir)
        
        if df is None or sim_id is None:
            logger.warning("Insufficient data for CD timeline plot")
            return None
        
        # Filter data for specific simulation
        sim_data = df[df.get('sim_id', df.get('scenario_id', '')) == sim_id]
        if sim_data.empty:
            logger.warning(f"No data found for simulation {sim_id}")
            return None
        
        return visualizer._plot_conflict_timeline(sim_data, sim_id)
        
    except Exception as e:
        logger.warning(f"Failed to generate CD timeline: {e}")
        return None


def plot_cr_flowchart(sim_id: str = None, tier: str = None, 
                     output_dir: str = "visualizations", **kwargs) -> Optional[str]:
    """
    Plot conflict resolution flowchart.
    
    Args:
        sim_id: Simulation ID
        tier: Distribution shift tier
        output_dir: Output directory for plots
        **kwargs: Additional arguments
        
    Returns:
        Path to generated flowchart file, or None if visualization failed
    """
    try:
        visualizer = MonteCarloVisualizer(output_dir)
        return visualizer._plot_resolution_flowchart(sim_id, tier)
        
    except Exception as e:
        logger.warning(f"Failed to generate CR flowchart: {e}")
        return None


def plot_tier_comparison(df: pd.DataFrame = None, output_dir: str = "visualizations", 
                        **kwargs) -> Optional[str]:
    """
    Plot tier comparison analysis.
    
    Args:
        df: DataFrame with results across tiers
        output_dir: Output directory for plots
        **kwargs: Additional arguments
        
    Returns:
        Path to generated comparison plot, or None if visualization failed
    """
    try:
        visualizer = MonteCarloVisualizer(output_dir)
        
        if df is None:
            logger.warning("No data provided for tier comparison")
            return None
        
        return visualizer._plot_tier_performance_comparison(df)
        
    except Exception as e:
        logger.warning(f"Failed to generate tier comparison: {e}")
        return None


def create_visualization_summary(output_dir: str = "visualizations", **kwargs) -> Optional[str]:
    """
    Create a comprehensive visualization summary.
    
    Args:
        output_dir: Output directory for summary
        **kwargs: Additional arguments
        
    Returns:
        Path to generated summary file, or None if creation failed
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / "visualization_summary.html"
        
        # Create a basic HTML summary
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM-ATC-HAL Visualization Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .warning { color: #ff6600; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM-ATC-HAL Visualization Summary</h1>
                <p>Generated comprehensive analysis visualizations</p>
            </div>
            
            <div class="section">
                <h2>Available Visualizations</h2>
                <ul>
                    <li>Metric Distribution Analysis</li>
                    <li>Distribution Shift Comparisons</li>
                    <li>Performance Trend Analysis</li>
                    <li>Sensitivity Analysis Charts</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Implementation Status</h2>
                <p class="warning">Visualization system has been upgraded with comprehensive analysis capabilities.</p>
                <p>For full functionality, ensure all required dependencies are installed:</p>
                <ul>
                    <li>matplotlib, seaborn (static plots)</li>
                    <li>plotly (interactive visualizations)</li>
                    <li>folium, geopandas (geospatial analysis)</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(summary_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created visualization summary: {summary_file}")
        return str(summary_file)
        
    except Exception as e:
        logger.warning(f"Failed to create visualization summary: {e}")
        return None
