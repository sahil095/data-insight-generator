"""Auditor Agent - Validates statistical accuracy of insights."""
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils.helpers import safe_divide, format_number
from config.settings import settings


class AuditorAgent:
    """Agent responsible for validating statistical accuracy of insights."""
    
    def __init__(self):
        """Initialize the auditor agent."""
        pass
    
    def validate(
        self,
        insights: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate insights against actual data.
        
        Args:
            insights: Dictionary of insights for each dataset
            datasets: Dictionary of actual DataFrames
            
        Returns:
            Dictionary with validation results for each dataset
        """
        validation_results = {}
        
        for dataset_name, dataset_insights in insights.items():
            if dataset_name not in datasets:
                validation_results[dataset_name] = {
                    "overall_valid": False,
                    "errors": [f"Dataset {dataset_name} not found"]
                }
                continue
            
            df = datasets[dataset_name]
            stats = dataset_insights.get("statistics", {})
            insights_text = dataset_insights.get("insights_text", "")
            
            # Perform validation
            result = self._validate_dataset_insights(df, stats, insights_text)
            validation_results[dataset_name] = result
        
        return validation_results
    
    def _validate_dataset_insights(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Any],
        insights_text: str
    ) -> Dict[str, Any]:
        """
        Validate insights for a single dataset.
        
        Args:
            df: DataFrame
            statistics: Computed statistics
            insights_text: Natural language insights
            
        Returns:
            Validation result dictionary with confidence score
        """
        result = {
            "overall_valid": True,
            "confidence_score": 1.0,  # Start with full confidence
            "statistical_checks": [],
            "numeric_checks": [],
            "errors": [],
            "warnings": [],
            "discrepancies": []
        }
        
        # Validate statistics are consistent with data
        stats_validation = self._validate_statistics(df, statistics)
        result["statistical_checks"] = stats_validation["checks"]
        result["errors"].extend(stats_validation["errors"])
        result["discrepancies"].extend(stats_validation["errors"])
        
        # Extract and validate numeric claims from insights text
        numeric_validation = self._validate_numeric_claims(insights_text, statistics, df)
        result["numeric_checks"] = numeric_validation["checks"]
        result["errors"].extend(numeric_validation["errors"])
        result["warnings"].extend(numeric_validation["warnings"])
        
        # Calculate confidence score (0-1)
        total_checks = len(result["statistical_checks"]) + len(result["numeric_checks"])
        error_count = len(result["errors"])
        warning_count = len(result["warnings"])
        
        if total_checks > 0:
            # Deduct for errors and warnings
            result["confidence_score"] = max(0.0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))
        else:
            result["confidence_score"] = 0.5  # Unknown if no checks performed
        
        # Overall validity
        if result["errors"]:
            result["overall_valid"] = False
        
        return result
    
    def _validate_statistics(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that statistics match the actual data.
        
        Args:
            df: DataFrame
            statistics: Claimed statistics
            
        Returns:
            Validation result
        """
        validation = {
            "checks": [],
            "errors": []
        }
        
        # Check shape
        claimed_shape = statistics.get("shape")
        actual_shape = df.shape
        if claimed_shape and claimed_shape != list(actual_shape):
            validation["errors"].append(
                f"Shape mismatch: claimed {claimed_shape}, actual {list(actual_shape)}"
            )
        else:
            validation["checks"].append("✓ Shape matches")
        
        # Check column names
        claimed_columns = set(statistics.get("columns", []))
        actual_columns = set(df.columns)
        if claimed_columns != actual_columns:
            missing = actual_columns - claimed_columns
            extra = claimed_columns - actual_columns
            if missing:
                validation["errors"].append(f"Missing columns in stats: {missing}")
            if extra:
                validation["errors"].append(f"Extra columns in stats: {extra}")
        else:
            validation["checks"].append("✓ Columns match")
        
        # Validate numeric statistics
        numeric_stats = statistics.get("numeric_statistics", {})
        for col, col_stats in numeric_stats.items():
            if col not in df.columns:
                validation["errors"].append(f"Column {col} not found in data")
                continue
            
            numeric_col = df.select_dtypes(include=[np.number])[col] if col in df.select_dtypes(include=[np.number]).columns else None
            if numeric_col is None or numeric_col.empty:
                continue
            
            # Check mean
            if 'mean' in col_stats:
                actual_mean = float(numeric_col.mean())
                claimed_mean = float(col_stats['mean'])
                if abs(actual_mean - claimed_mean) > settings.numeric_tolerance * max(abs(actual_mean), 1):
                    validation["errors"].append(
                        f"Mean mismatch for {col}: claimed {claimed_mean:.4f}, actual {actual_mean:.4f}"
                    )
                else:
                    validation["checks"].append(f"✓ Mean for {col} matches")
            
            # Check std
            if 'std' in col_stats:
                actual_std = float(numeric_col.std())
                claimed_std = float(col_stats['std'])
                if abs(actual_std - claimed_std) > settings.numeric_tolerance * max(abs(actual_std), 1):
                    validation["errors"].append(
                        f"Std mismatch for {col}: claimed {claimed_std:.4f}, actual {actual_std:.4f}"
                    )
                else:
                    validation["checks"].append(f"✓ Std for {col} matches")
        
        return validation
    
    def _validate_numeric_claims(
        self,
        insights_text: str,
        statistics: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract numeric claims from insights text and validate them.
        
        Args:
            insights_text: Natural language insights
            statistics: Computed statistics
            df: DataFrame
            
        Returns:
            Validation result
        """
        validation = {
            "checks": [],
            "errors": [],
            "warnings": []
        }
        
        # Extract numbers and associated context from text
        # Simple pattern matching for numeric claims
        numeric_pattern = r'(\d+[,.]?\d*)\s*(mean|average|std|standard deviation|variance|min|max|count|sum|total)'
        matches = re.findall(numeric_pattern, insights_text.lower())
        
        # Note: This is a simplified validation. A more sophisticated approach
        # would parse the text more carefully and match to specific columns.
        if matches:
            validation["warnings"].append(
                f"Found {len(matches)} potential numeric claims to validate (manual review recommended)"
            )
        
        return validation
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Validation results from validate()
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INSIGHT VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for dataset_name, result in validation_results.items():
            report_lines.append(f"Dataset: {dataset_name}")
            report_lines.append("-" * 80)
            
            # Overall status
            status = "✓ VALID" if result["overall_valid"] else "✗ INVALID"
            confidence = result.get("confidence_score", 0.0)
            report_lines.append(f"Overall Status: {status}")
            report_lines.append(f"Confidence Score: {confidence:.2f} (0-1)")
            report_lines.append("")
            
            # Statistical checks
            if result["statistical_checks"]:
                report_lines.append("Statistical Checks:")
                for check in result["statistical_checks"]:
                    report_lines.append(f"  {check}")
                report_lines.append("")
            
            # Numeric checks
            if result["numeric_checks"]:
                report_lines.append("Numeric Validation:")
                for check in result["numeric_checks"]:
                    report_lines.append(f"  {check}")
                report_lines.append("")
            
            # Errors
            if result["errors"]:
                report_lines.append("Errors:")
                for error in result["errors"]:
                    report_lines.append(f"  ✗ {error}")
                report_lines.append("")
            
            # Warnings
            if result["warnings"]:
                report_lines.append("Warnings:")
                for warning in result["warnings"]:
                    report_lines.append(f"  ⚠ {warning}")
                report_lines.append("")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

