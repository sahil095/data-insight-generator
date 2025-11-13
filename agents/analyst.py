"""Analyst Agent - Performs data analysis and generates insights."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from tools.visualization import VisualizationTool
from utils.helpers import format_number, get_data_summary
from utils.llm_client import UnifiedLLMClient
from config.settings import settings


class AnalystAgent:
    """Agent responsible for analyzing data and generating insights."""
    
    def __init__(self):
        """Initialize the analyst agent."""
        try:
            self.client = UnifiedLLMClient()
        except Exception as e:
            print(f"Warning: LLM client initialization failed: {e}")
            self.client = None
        self.visualization_tool = VisualizationTool()
    
    def analyze(
        self,
        datasets: Dict[str, pd.DataFrame],
        generate_visualizations: bool = True,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze datasets and generate insights.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            generate_visualizations: Whether to generate visualizations
            query: Optional user query to guide analysis
            
        Returns:
            Dictionary with insights for each dataset
        """
        all_insights = {}
        
        for dataset_name, df in datasets.items():
            print(f"  Analyzing {dataset_name}...")
            
            # Compute statistics
            statistics = self._compute_statistics(df)
            
            # Generate visualizations
            visualization_paths = {}
            if generate_visualizations:
                try:
                    plots = self.visualization_tool.generate_summary_plots(df, query=query)
                    visualization_paths = {k: str(v) for k, v in plots.items()}
                except Exception as e:
                    print(f"  Warning: Visualization generation failed: {e}")
            
            # Generate natural language insights
            insights_text = self._generate_insights(df, statistics)
            
            all_insights[dataset_name] = {
                "statistics": statistics,
                "insights_text": insights_text,
                "visualizations": visualization_paths,
                "data_summary": get_data_summary(df)
            }
        
        return all_insights
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive statistical measures for a DataFrame (Enhanced EDA).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with computed statistics
        """
        # Basic EDA checks
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        
        # NULL/Missing value analysis
        null_counts = df.isnull().sum().to_dict()
        null_percentages = {col: (count / total_rows * 100) if total_rows > 0 else 0 
                           for col, count in null_counts.items()}
        total_nulls = sum(null_counts.values())
        null_percentage_total = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        
        stats = {
            "shape": df.shape,
            "row_count": int(total_rows),
            "column_count": int(total_cols),
            "total_cells": int(total_cells),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": null_counts,
            "null_percentages": null_percentages,
            "total_nulls": int(total_nulls),
            "null_percentage_total": float(null_percentage_total),
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": float(duplicate_percentage),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "numeric_statistics": {}
        }
        
        # Compute numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats["numeric_statistics"] = numeric_stats
            
            # Additional statistics for each numeric column
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats["numeric_statistics"][col].update({
                        "variance": float(df[col].var()) if len(col_data) > 1 else None,
                        "skewness": float(df[col].skew()) if len(col_data) > 2 else None,
                        "kurtosis": float(df[col].kurtosis()) if len(col_data) > 2 else None,
                        "median": float(df[col].median()),
                        "iqr": float(df[col].quantile(0.75) - df[col].quantile(0.25)),
                        "null_count": int(null_counts.get(col, 0)),
                        "null_percentage": float(null_percentages.get(col, 0)),
                        "zeros": int((df[col] == 0).sum()),
                        "negatives": int((df[col] < 0).sum()),
                        "positives": int((df[col] > 0).sum()),
                    })
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            stats["categorical_statistics"] = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts().to_dict()
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
                stats["categorical_statistics"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "mode": str(mode_value) if mode_value is not None else None,
                    "mode_count": int(value_counts.get(mode_value, 0)) if mode_value is not None else 0,
                    "top_values": dict(list(value_counts.items())[:10]),
                    "null_count": int(null_counts.get(col, 0)),
                    "null_percentage": float(null_percentages.get(col, 0)),
                }
        
        # Correlation matrix (if numeric columns exist)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().to_dict()
            stats["correlations"] = corr_matrix
        
        # Data quality summary
        stats["data_quality"] = {
            "completeness": float(100 - null_percentage_total),
            "uniqueness": float(100 - duplicate_percentage),
            "has_nulls": total_nulls > 0,
            "has_duplicates": duplicate_count > 0,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
        }
        
        return stats
    
    def _generate_insights(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Any]
    ) -> str:
        """
        Generate natural language insights from data and statistics.
        
        Args:
            df: DataFrame
            statistics: Computed statistics
            
        Returns:
            Natural language insights text
        """
        if not self.client:
            # Fallback to template-based insights if OpenAI not available
            return self._generate_template_insights(df, statistics)
        
        try:
            # Prepare context for LLM
            context = {
                "shape": statistics["shape"],
                "columns": statistics["columns"],
                "null_counts": statistics["null_counts"],
                "numeric_statistics": statistics.get("numeric_statistics", {}),
                "categorical_statistics": statistics.get("categorical_statistics", {})
            }
            
            prompt = f"""Analyze the following dataset statistics and generate clear, concise insights in natural language.

Dataset Statistics:
{json.dumps(context, indent=2)}

Generate insights that:
1. Summarize the dataset structure (rows, columns, data types)
2. Highlight key patterns in numeric columns (means, distributions, outliers)
3. Identify important categorical patterns
4. Note any data quality issues (missing values, etc.)
5. Provide actionable observations

Format your response as clear, well-structured paragraphs."""

            response = self.client.chat_completions_create(
                messages=[
                    {"role": "system", "content": "You are a data analyst expert at generating clear, accurate insights from dataset statistics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Warning: LLM insight generation failed: {e}")
            return self._generate_template_insights(df, statistics)
    
    def _generate_template_insights(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Any]
    ) -> str:
        """
        Generate template-based insights (fallback).
        
        Args:
            df: DataFrame
            statistics: Computed statistics
            
        Returns:
            Template-based insights text
        """
        insights = []
        
        # Dataset overview
        insights.append(f"The dataset contains {statistics['shape'][0]:,} rows and {statistics['shape'][1]} columns.")
        
        # Numeric insights
        numeric_stats = statistics.get("numeric_statistics", {})
        if numeric_stats:
            insights.append("\nNumeric Column Analysis:")
            for col, stats_col in numeric_stats.items():
                if 'mean' in stats_col:
                    mean = stats_col['mean']
                    std = stats_col.get('std', 0)
                    insights.append(
                        f"  - {col}: Mean = {format_number(mean)}, "
                        f"Std = {format_number(std)}, "
                        f"Range = [{format_number(stats_col.get('min', 0))}, {format_number(stats_col.get('max', 0))}]"
                    )
        
        # Categorical insights
        cat_stats = statistics.get("categorical_statistics", {})
        if cat_stats:
            insights.append("\nCategorical Column Analysis:")
            for col, stats_col in cat_stats.items():
                insights.append(
                    f"  - {col}: {stats_col['unique_count']} unique values, "
                    f"most common = {stats_col.get('mode', 'N/A')}"
                )
        
        # Data quality
        null_counts = statistics.get("null_counts", {})
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            insights.append(f"\nData Quality: {total_nulls:,} missing values across the dataset.")
        
        return "\n".join(insights)
    
    def save_insights(self, insights: Dict[str, Any], filepath: str) -> None:
        """
        Save insights to a JSON file.
        
        Args:
            insights: Insights dictionary
            filepath: Path to save file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        serializable_insights = convert_to_serializable(insights)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_insights, f, indent=2)

