"""Block H: Statistical Analysis for trill /r/ sociolinguistic variation."""

from .base import AnalysisConfig, TestResult, AnalysisResult, interpret_effect_size
from .data_loader import load_analysis_data, merge_with_metadata
from .descriptive import compute_summary_stats, compute_all_descriptive_stats, get_group_counts
from .inferential import run_two_group_test, run_multi_group_test, run_all_tests
from .visualization import create_boxplots, create_violin_plots, create_all_visualizations
from .report_generator import generate_report, save_tables

__all__ = [
    # Base
    'AnalysisConfig',
    'TestResult',
    'AnalysisResult',
    'interpret_effect_size',
    # Data loading
    'load_analysis_data',
    'merge_with_metadata',
    # Descriptive
    'compute_summary_stats',
    'compute_all_descriptive_stats',
    'get_group_counts',
    # Inferential
    'run_two_group_test',
    'run_multi_group_test',
    'run_all_tests',
    # Visualization
    'create_boxplots',
    'create_violin_plots',
    'create_all_visualizations',
    # Reports
    'generate_report',
    'save_tables',
]
