"""Base classes and data types for statistical analysis (Block H)."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""

    # Outcome variables to analyze
    outcome_variables: List[str] = field(default_factory=lambda: [
        'num_cycles', 'duration_ms', 'voicing_pct',
        'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db'
    ])

    # Grouping variables (predictors)
    predictor_variables: List[str] = field(default_factory=lambda: [
        'sex', 'age_bin', 'education_bin', 'country', 'context_label'
    ])

    # Statistical thresholds
    alpha: float = 0.05
    min_group_size: int = 10  # Minimum n per group for analysis

    # Multiple comparison correction
    correction_method: str = 'bonferroni'  # or 'fdr_bh', 'holm'

    # Visualization settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150


@dataclass
class TestResult:
    """Result of a single statistical test."""

    test_id: str
    outcome_variable: str
    predictor_variable: str
    test_type: str

    statistic: float
    p_value: float
    p_adjusted: Optional[float] = None

    effect_size: Optional[float] = None
    effect_size_name: str = ''
    effect_size_interpretation: str = ''  # small, medium, large

    group_ns: Dict[str, int] = field(default_factory=dict)
    group_means: Dict[str, float] = field(default_factory=dict)
    group_medians: Dict[str, float] = field(default_factory=dict)

    significant: bool = False
    notes: str = ''

    posthoc_results: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'test_id': self.test_id,
            'outcome_variable': self.outcome_variable,
            'predictor_variable': self.predictor_variable,
            'test_type': self.test_type,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'p_adjusted': self.p_adjusted,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'effect_size_interpretation': self.effect_size_interpretation,
            'significant': self.significant,
            'notes': self.notes,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    config: AnalysisConfig
    descriptive_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_results: List[TestResult] = field(default_factory=list)

    figures: List[str] = field(default_factory=list)  # Paths to generated figures
    issues: List[str] = field(default_factory=list)

    def add_test(self, test: TestResult) -> None:
        """Add a test result."""
        self.test_results.append(test)

    def add_issue(self, issue: str) -> None:
        """Log an issue."""
        self.issues.append(issue)

    def get_tests_df(self) -> pd.DataFrame:
        """Get test results as DataFrame."""
        if not self.test_results:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.test_results])

    def get_significant_tests(self) -> List[TestResult]:
        """Get only significant test results."""
        return [t for t in self.test_results if t.significant]


def interpret_effect_size(effect_size: float, metric: str = 'cohens_d') -> str:
    """
    Interpret effect size magnitude.

    Args:
        effect_size: The effect size value (absolute)
        metric: Type of effect size metric

    Returns:
        Interpretation string: 'negligible', 'small', 'medium', 'large'
    """
    es = abs(effect_size)

    if metric in ('cohens_d', 'rank_biserial'):
        # Cohen's d thresholds
        if es < 0.2:
            return 'negligible'
        elif es < 0.5:
            return 'small'
        elif es < 0.8:
            return 'medium'
        else:
            return 'large'

    elif metric in ('eta_squared', 'epsilon_squared'):
        # Eta-squared thresholds
        if es < 0.01:
            return 'negligible'
        elif es < 0.06:
            return 'small'
        elif es < 0.14:
            return 'medium'
        else:
            return 'large'

    else:
        # Default thresholds
        if es < 0.1:
            return 'negligible'
        elif es < 0.3:
            return 'small'
        elif es < 0.5:
            return 'medium'
        else:
            return 'large'
