"""Analyst Agent - Performs data analysis and generates insights."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from tools.visualization import VisualizationTool
from utils.helpers import format_number, get_data_summary
from config.settings import settings


class AnalystAgent:
    """Agent responsible for analyzing data and generating insights."""
    
    def __init__(self):
        """Initialize the analyst agent."""
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.visualization_tool = VisualizationTool()
    
    def analyze(
        self,
        datasets: Dict[str, pd.DataFrame],
        generate_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze datasets and generate insights.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            generate_visualizations: Whether to generate visualizations
            
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
                    plots = self.visualization_tool.generate_summary_plots(df)
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
        Compute statistical measures for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with computed statistics
        """
        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_statistics": {}
        }
        
        # Compute numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats["numeric_statistics"] = numeric_stats
            
            # Additional statistics
            for col in numeric_cols:
                stats["numeric_statistics"][col].update({
                    "variance": float(df[col].var()) if not df[col].empty else None,
                    "skewness": float(df[col].skew()) if len(df[col]) > 2 else None,
                    "kurtosis": float(df[col].kurtosis()) if len(df[col]) > 2 else None,
                })
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats["categorical_statistics"] = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts().to_dict()
                stats["categorical_statistics"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                    "top_values": dict(list(value_counts.items())[:10])
                }
        
        # Correlation matrix (if numeric columns exist)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().to_dict()
            stats["correlations"] = corr_matrix
        
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

            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are a data analyst expert at generating clear, accurate insights from dataset statistics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.llm_temperature,
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
        
        with open(filepath, 'w') as f:
            json.dump(serializable_insights, f, indent=2)

