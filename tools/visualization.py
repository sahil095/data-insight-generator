"""Visualization utilities for data analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config.settings import settings


class VisualizationTool:
    """Tool for generating data visualizations."""
    
    def __init__(self):
        """Initialize visualization tool."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
    def generate_distribution_plot(
        self,
        data: pd.Series,
        title: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a distribution plot for a numeric series.
        
        Args:
            data: Data series to plot
            title: Plot title
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(data.dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'{title} - Distribution')
        axes[0].set_xlabel(title)
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(data.dropna())
        axes[1].set_title(f'{title} - Box Plot')
        axes[1].set_ylabel(title)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}_distribution.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame with numeric columns
            title: Plot title
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation")
        
        # Compute correlation
        corr = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_scatter_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Scatter Matrix",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a scatter plot matrix for numeric columns.
        
        Args:
            df: DataFrame
            columns: List of columns to include (default: all numeric)
            title: Plot title
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
            columns = numeric_df.columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 numeric columns for scatter matrix")
        
        # Limit to top 6 columns to avoid overcrowding
        if len(columns) > 6:
            columns = columns[:6]
        
        pd.plotting.scatter_matrix(
            df[columns],
            figsize=(15, 15),
            alpha=0.5,
            diagonal='kde'
        )
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_time_series_plot(
        self,
        data: pd.Series,
        title: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a time series plot.
        
        Args:
            data: Time series data
            title: Plot title
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(14, 6))
        plt.plot(data.index, data.values, linewidth=2)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}_timeseries.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_pie_chart(
        self,
        data: pd.Series,
        title: str = "Pie Chart",
        top_n: int = 10,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a pie chart for categorical data.
        
        Args:
            data: Data series to plot
            title: Plot title
            top_n: Number of top categories to show (others grouped as 'Other')
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        value_counts = data.value_counts()
        
        # Take top N and group rest as 'Other'
        if len(value_counts) > top_n:
            top_values = value_counts.head(top_n)
            other_count = value_counts.iloc[top_n:].sum()
            top_values['Other'] = other_count
        else:
            top_values = value_counts
        
        plt.figure(figsize=(10, 8))
        plt.pie(
            top_values.values,
            labels=top_values.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}_pie.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Scatter Plot",
        hue_col: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a scatter plot for two numeric columns.
        
        Args:
            df: DataFrame
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Plot title
            hue_col: Optional column for color coding
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))
        
        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6, s=50)
        else:
            plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
        
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}_scatter.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_box_plot(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Box Plot",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate box plots for numeric columns.
        
        Args:
            df: DataFrame
            columns: List of columns to plot (default: all numeric)
            title: Plot title
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            raise ValueError("No numeric columns found for box plot")
        
        # Limit to 8 columns for readability
        if len(columns) > 8:
            columns = columns[:8]
        
        fig, ax = plt.subplots(figsize=(max(12, len(columns) * 1.5), 6))
        df[columns].boxplot(ax=ax, rot=45)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path is None:
            output_path = settings.output_dir / "visualizations" / f"{title.replace(' ', '_')}_box.{settings.visualization_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=settings.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_summary_plots(
        self,
        df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        query: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Generate a comprehensive set of summary plots for a DataFrame.
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save plots (optional)
            query: Optional user query to guide visualization selection
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = settings.output_dir / "visualizations"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Correlation heatmap for numeric columns
        if len(numeric_cols) >= 2:
            try:
                plots['correlation'] = self.generate_correlation_heatmap(
                    df,
                    output_path=output_dir / f"correlation_heatmap.{settings.visualization_format}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate correlation heatmap: {e}")
        
        # Distribution plots for each numeric column (limit to 6)
        for col in numeric_cols[:6]:
            try:
                plots[f'distribution_{col}'] = self.generate_distribution_plot(
                    df[col],
                    title=col,
                    output_path=output_dir / f"distribution_{col}.{settings.visualization_format}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate distribution plot for {col}: {e}")
        
        # Box plots for numeric columns
        if len(numeric_cols) > 0:
            try:
                plots['box_plot'] = self.generate_box_plot(
                    df,
                    columns=numeric_cols[:8],
                    output_path=output_dir / f"box_plot.{settings.visualization_format}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate box plot: {e}")
        
        # Pie charts for categorical columns (limit to 3)
        for col in categorical_cols[:3]:
            try:
                if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                    plots[f'pie_{col}'] = self.generate_pie_chart(
                        df[col],
                        title=f"{col} Distribution",
                        output_path=output_dir / f"pie_{col}.{settings.visualization_format}"
                    )
            except Exception as e:
                print(f"Warning: Failed to generate pie chart for {col}: {e}")
        
        # Scatter plots for numeric pairs (if query suggests relationships)
        if len(numeric_cols) >= 2:
            try:
                # Generate scatter plot for first two numeric columns
                plots['scatter_plot'] = self.generate_scatter_plot(
                    df,
                    x_col=numeric_cols[0],
                    y_col=numeric_cols[1],
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                    output_path=output_dir / f"scatter_{numeric_cols[0]}_vs_{numeric_cols[1]}.{settings.visualization_format}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate scatter plot: {e}")
        
        return plots

