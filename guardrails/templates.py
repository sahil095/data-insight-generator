"""Prompt templates with reasoning chains for validation."""
from typing import Dict, Any


class PromptTemplates:
    """Templates for prompts with explicit reasoning chains."""
    
    @staticmethod
    def get_analysis_template() -> str:
        """Get template for data analysis prompt."""
        return """Analyze the following dataset and generate insights. For each statistical claim you make, ensure you:

1. Calculate the actual value from the data
2. State the computed value explicitly
3. Provide context for interpretation

Dataset Statistics:
{statistics}

Instructions:
- Compute all statistics explicitly before stating them
- Show your work for key calculations
- Verify numerical accuracy
- Provide clear, actionable insights

Generate your analysis:"""

    @staticmethod
    def get_validation_template() -> str:
        """Get template for validation prompt with reasoning chain."""
        return """Validate the following statistical claims. For each claim:

1. Extract the stated value
2. Compute the actual value from the data
3. Compare the two values
4. Determine if they match within tolerance ({tolerance})
5. Report the result

Claims to validate:
{claims}

Dataset:
{dataset_summary}

Provide validation with explicit calculations:"""

    @staticmethod
    def get_insight_generation_template() -> str:
        """Get template for insight generation with reasoning."""
        return """Generate insights from this dataset. For each insight:

REASONING CHAIN:
1. What pattern or statistic are you identifying?
2. What is the computed value? (Show calculation)
3. What does this value mean in context?
4. Why is this insight useful?

Dataset:
- Shape: {shape}
- Columns: {columns}
- Statistics: {statistics}

Generate insights following the reasoning chain above:"""

    @staticmethod
    def format_statistics_for_prompt(statistics: Dict[str, Any]) -> str:
        """
        Format statistics dictionary for use in prompts.
        
        Args:
            statistics: Statistics dictionary
            
        Returns:
            Formatted string
        """
        lines = []
        
        if "shape" in statistics:
            lines.append(f"Shape: {statistics['shape']}")
        
        if "columns" in statistics:
            lines.append(f"Columns: {', '.join(statistics['columns'])}")
        
        if "numeric_statistics" in statistics:
            lines.append("\nNumeric Statistics:")
            for col, stats in statistics["numeric_statistics"].items():
                lines.append(f"  {col}:")
                if "mean" in stats:
                    lines.append(f"    Mean: {stats['mean']:.4f}")
                if "std" in stats:
                    lines.append(f"    Std: {stats['std']:.4f}")
                if "min" in stats:
                    lines.append(f"    Min: {stats['min']:.4f}")
                if "max" in stats:
                    lines.append(f"    Max: {stats['max']:.4f}")
        
        if "categorical_statistics" in statistics:
            lines.append("\nCategorical Statistics:")
            for col, stats in statistics["categorical_statistics"].items():
                lines.append(f"  {col}:")
                lines.append(f"    Unique values: {stats.get('unique_count', 0)}")
                if "mode" in stats:
                    lines.append(f"    Mode: {stats['mode']}")
        
        return "\n".join(lines)

