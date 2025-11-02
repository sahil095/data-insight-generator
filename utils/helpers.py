"""Utility helper functions."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
        
    Returns:
        The division result or default value
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def format_number(value: float, decimals: int = 2, use_scientific: bool = False) -> str:
    """
    Format a number for display.
    
    Args:
        value: The number to format
        decimals: Number of decimal places
        use_scientific: Whether to use scientific notation for large numbers
        
    Returns:
        Formatted string
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    
    if use_scientific and abs(value) >= 10000:
        return f"{value:.{decimals}e}"
    
    return f"{value:,.{decimals}f}".rstrip("0").rstrip(".")


def clean_column_name(name: str) -> str:
    """
    Clean a column name by removing special characters and normalizing.
    
    Args:
        name: Original column name
        
    Returns:
        Cleaned column name
    """
    # Replace spaces with underscores and convert to lowercase
    cleaned = name.strip().replace(" ", "_").lower()
    # Remove special characters, keep only alphanumeric and underscores
    cleaned = "".join(c if c.isalnum() or c == "_" else "" for c in cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed_column"


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate a DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required column names
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    if df is None:
        result["valid"] = False
        result["errors"].append("DataFrame is None")
        return result
    
    if df.empty:
        result["valid"] = False
        result["errors"].append("DataFrame is empty")
        return result
    
    if len(df) == 0:
        result["warnings"].append("DataFrame has no rows")
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {missing}")
    
    return result


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of DataFrame statistics.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_stats": {}
    }
    
    # Add numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return summary

