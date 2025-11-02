"""Numeric validator for checking statistical correctness."""
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from utils.helpers import safe_divide
from config.settings import settings


class NumericValidator:
    """Validator for numeric correctness of insights."""
    
    def __init__(self):
        """Initialize the numeric validator."""
        pass
    
    def validate_insight_numerics(
        self,
        insights_text: str,
        statistics: Dict[str, Dict[str, Any]],
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate numeric claims in insights text against actual data.
        
        Args:
            insights_text: Natural language insights
            statistics: Computed statistics
            datasets: Actual DataFrames
            
        Returns:
            Validation result with accuracy metrics
        """
        result = {
            "total_claims": 0,
            "valid_claims": 0,
            "invalid_claims": 0,
            "accuracy": 0.0,
            "claims": [],
            "errors": []
        }
        
        # Extract numeric claims from text
        claims = self._extract_numeric_claims(insights_text)
        result["total_claims"] = len(claims)
        
        if len(claims) == 0:
            result["accuracy"] = 1.0  # No claims to validate
            return result
        
        # Validate each claim
        for claim in claims:
            validation = self._validate_claim(claim, statistics, datasets)
            result["claims"].append(validation)
            
            if validation["valid"]:
                result["valid_claims"] += 1
            else:
                result["invalid_claims"] += 1
        
        # Calculate accuracy
        result["accuracy"] = safe_divide(
            result["valid_claims"],
            result["total_claims"],
            default=1.0
        )
        
        return result
    
    def _extract_numeric_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric claims from text.
        
        Args:
            text: Natural language text
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Pattern for numbers with context
        # This is a simplified extraction - could be improved with NLP
        patterns = [
            # Mean/average patterns
            (r'(?:mean|average|avg)[:\s]*([+-]?\d+\.?\d*)', 'mean'),
            # Standard deviation patterns
            (r'(?:std|standard deviation)[:\s]*([+-]?\d+\.?\d*)', 'std'),
            # Count patterns
            (r'(?:count|total)[:\s]*(\d+)', 'count'),
            # Percentage patterns
            (r'(\d+\.?\d*)%', 'percentage'),
        ]
        
        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    claims.append({
                        "type": claim_type,
                        "value": value,
                        "context": match.group(0),
                        "text": text[max(0, match.start()-50):match.end()+50]
                    })
                except (ValueError, IndexError):
                    continue
        
        return claims
    
    def _validate_claim(
        self,
        claim: Dict[str, Any],
        statistics: Dict[str, Dict[str, Any]],
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate a single numeric claim.
        
        Args:
            claim: Claim dictionary
            statistics: Computed statistics
            datasets: DataFrames
            
        Returns:
            Validation result
        """
        claim_type = claim["type"]
        claimed_value = claim["value"]
        
        # Try to find matching statistic
        validation_result = {
            "claim": claim,
            "valid": False,
            "computed_value": None,
            "error": None,
            "matched_column": None
        }
        
        # Search through statistics to find matching value
        for dataset_name, dataset_stats in statistics.items():
            numeric_stats = dataset_stats.get("numeric_statistics", {})
            
            for col_name, col_stats in numeric_stats.items():
                if claim_type == "mean" and "mean" in col_stats:
                    computed = float(col_stats["mean"])
                    if self._values_match(claimed_value, computed):
                        validation_result["valid"] = True
                        validation_result["computed_value"] = computed
                        validation_result["matched_column"] = f"{dataset_name}:{col_name}"
                        return validation_result
                
                elif claim_type == "std" and "std" in col_stats:
                    computed = float(col_stats["std"])
                    if self._values_match(claimed_value, computed):
                        validation_result["valid"] = True
                        validation_result["computed_value"] = computed
                        validation_result["matched_column"] = f"{dataset_name}:{col_name}"
                        return validation_result
        
        # If no match found, try to compute from data
        if datasets and claim_type in ["mean", "std"]:
            # Try to find column that might match
            for dataset_name, df in datasets.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if claim_type == "mean":
                        computed = float(df[col].mean())
                    elif claim_type == "std":
                        computed = float(df[col].std())
                    else:
                        continue
                    
                    if self._values_match(claimed_value, computed):
                        validation_result["valid"] = True
                        validation_result["computed_value"] = computed
                        validation_result["matched_column"] = f"{dataset_name}:{col}"
                        return validation_result
        
        # No match found
        validation_result["error"] = f"Could not find matching statistic for {claim_type} = {claimed_value}"
        return validation_result
    
    def _values_match(self, value1: float, value2: float, tolerance: float = None) -> bool:
        """
        Check if two numeric values match within tolerance.
        
        Args:
            value1: First value
            value2: Second value
            tolerance: Acceptable tolerance (default: settings.numeric_tolerance)
            
        Returns:
            True if values match
        """
        if tolerance is None:
            tolerance = settings.numeric_tolerance
        
        error = abs(value1 - value2)
        relative_error = safe_divide(error, max(abs(value1), abs(value2), 1))
        return relative_error <= tolerance
    
    def generate_numeric_report(self, validation_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable report of numeric validation.
        
        Args:
            validation_result: Result from validate_insight_numerics()
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("NUMERIC VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Claims Found: {validation_result['total_claims']}")
        lines.append(f"Valid Claims: {validation_result['valid_claims']}")
        lines.append(f"Invalid Claims: {validation_result['invalid_claims']}")
        lines.append(f"Accuracy: {validation_result['accuracy']*100:.1f}%")
        lines.append("")
        
        if validation_result['claims']:
            lines.append("Claim Details:")
            lines.append("-" * 80)
            
            for i, claim_result in enumerate(validation_result['claims'], 1):
                claim = claim_result['claim']
                status = "✓" if claim_result['valid'] else "✗"
                
                lines.append(f"{i}. {status} {claim['type'].upper()}: {claim['value']}")
                lines.append(f"   Context: {claim['context']}")
                
                if claim_result['valid']:
                    lines.append(f"   ✓ Matched: {claim_result['computed_value']} ({claim_result['matched_column']})")
                else:
                    lines.append(f"   ✗ {claim_result.get('error', 'No match found')}")
                
                lines.append("")
        
        if validation_result['errors']:
            lines.append("Errors:")
            for error in validation_result['errors']:
                lines.append(f"  - {error}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

