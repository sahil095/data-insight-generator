"""Data quality validators and guardrails."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils.helpers import safe_divide
from config.settings import settings


class DataQualityGuardrail:
    """Guardrails for data quality validation."""
    
    def __init__(self):
        """Initialize the data quality guardrail."""
        pass
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality for a DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics and issues
        """
        result = {
            "valid": True,
            "score": 0.0,
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        if df is None or df.empty:
            result["valid"] = False
            result["score"] = 0.0
            result["issues"].append("DataFrame is empty or None")
            return result
        
        metrics = {}
        issues = []
        warnings = []
        
        # Check 1: Null values
        null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        metrics["null_percentage"] = null_percentage
        if null_percentage > 50:
            result["valid"] = False
            issues.append(f"High null percentage: {null_percentage:.2f}%")
        elif null_percentage > 20:
            warnings.append(f"Moderate null percentage: {null_percentage:.2f}%")
        
        # Check 2: Duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        metrics["duplicate_percentage"] = duplicate_percentage
        if duplicate_percentage > 10:
            warnings.append(f"High duplicate rows: {duplicate_percentage:.2f}%")
        
        # Check 3: Column consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Column {col} has {inf_count} infinite values")
            
            # Check for extreme outliers (beyond 5 standard deviations)
            if len(df[col].dropna()) > 0:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = ((df[col] - mean).abs() > 5 * std).sum()
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        warnings.append(f"Column {col} has {outliers} potential outliers")
        
        # Check 4: Data types
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Check if object column might be numeric
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                warnings.append(f"Column {col} appears numeric but is stored as object")
            except (ValueError, TypeError):
                pass
        
        # Check 5: Minimum data size
        if len(df) < 10:
            warnings.append(f"Small dataset size: {len(df)} rows")
        
        # Calculate quality score
        score = 1.0
        
        # Deduct for null values
        score -= min(null_percentage / 100, 0.5)
        
        # Deduct for duplicates
        score -= min(duplicate_percentage / 50, 0.2)
        
        # Deduct for issues
        score -= min(len(issues) * 0.1, 0.3)
        
        score = max(0.0, score)
        
        result["metrics"] = metrics
        result["issues"] = issues
        result["warnings"] = warnings
        result["score"] = score
        result["valid"] = score >= settings.min_data_quality_score
        
        return result
    
    def validate_statistical_claim(
        self,
        claim: str,
        computed_value: float,
        claimed_value: float,
        tolerance: float = None
    ) -> Dict[str, Any]:
        """
        Validate a statistical claim against computed value.
        
        Args:
            claim: Description of the claim
            computed_value: Actually computed value
            claimed_value: Value stated in claim
            tolerance: Acceptable tolerance (default: settings.numeric_tolerance)
            
        Returns:
            Validation result
        """
        if tolerance is None:
            tolerance = settings.numeric_tolerance
        
        error = abs(computed_value - claimed_value)
        relative_error = safe_divide(error, max(abs(computed_value), 1))
        
        is_valid = relative_error <= tolerance
        
        return {
            "valid": is_valid,
            "claim": claim,
            "computed": computed_value,
            "claimed": claimed_value,
            "error": error,
            "relative_error": relative_error,
            "tolerance": tolerance
        }

