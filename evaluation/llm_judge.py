"""LLM-as-a-judge evaluation for insight quality."""
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from config.settings import settings


class LLMJudge:
    """LLM-based judge for evaluating insight quality."""
    
    def __init__(self):
        """Initialize the LLM judge."""
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        if not self.client:
            print("Warning: OpenAI API key not set. LLM judge evaluation will be limited.")
    
    def evaluate_insight_quality(
        self,
        insights: Dict[str, Any],
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of generated insights using LLM-as-a-judge.
        
        Args:
            insights: Dictionary with insights for a dataset
            data_context: Context about the data (shape, columns, etc.)
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        if not self.client:
            # Return default evaluation if LLM not available
            return {
                "clarity_evaluation": {
                    "overall_score": 0.5,
                    "feedback": "LLM evaluation unavailable - API key not configured"
                },
                "relevance_evaluation": {
                    "overall_score": 0.5,
                    "feedback": "LLM evaluation unavailable - API key not configured"
                }
            }
        
        insights_text = insights.get("insights_text", "")
        statistics = insights.get("statistics", {})
        
        # Evaluate clarity
        clarity_eval = self._evaluate_clarity(insights_text, data_context)
        
        # Evaluate relevance
        relevance_eval = self._evaluate_relevance(insights_text, statistics, data_context)
        
        return {
            "clarity_evaluation": clarity_eval,
            "relevance_evaluation": relevance_eval
        }
    
    def _evaluate_clarity(
        self,
        insights_text: str,
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the clarity of insights.
        
        Args:
            insights_text: Natural language insights
            data_context: Context about the data
            
        Returns:
            Clarity evaluation result
        """
        if not self.client:
            return {"overall_score": 0.5, "feedback": "LLM unavailable"}
        
        prompt = f"""Evaluate the clarity of the following data insights. Rate on a scale of 0-1 where:
- 1.0 = Exceptionally clear, well-structured, easy to understand
- 0.7 = Clear with minor issues
- 0.5 = Somewhat clear but needs improvement
- 0.3 = Unclear or confusing
- 0.0 = Very unclear or incomprehensible

Data Context:
- Shape: {data_context.get('shape', 'unknown')}
- Columns: {data_context.get('columns', [])}

Insights:
{insights_text}

Provide:
1. An overall clarity score (0-1)
2. Specific feedback on what makes it clear or unclear
3. Suggestions for improvement

Respond in JSON format:
{{
    "overall_score": <float 0-1>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of data analysis insights. Provide clear, structured feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                response_format={"type": "json_object"} if hasattr(self.client.chat.completions.create, "__code__") else None
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                eval_result = json.loads(content)
                return {
                    "overall_score": float(eval_result.get("overall_score", 0.5)),
                    "feedback": content,
                    "strengths": eval_result.get("strengths", []),
                    "weaknesses": eval_result.get("weaknesses", []),
                    "suggestions": eval_result.get("suggestions", [])
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, extract score from text
                return self._extract_score_from_text(content)
                
        except Exception as e:
            print(f"Warning: LLM clarity evaluation failed: {e}")
            return {
                "overall_score": 0.5,
                "feedback": f"Evaluation failed: {str(e)}"
            }
    
    def _evaluate_relevance(
        self,
        insights_text: str,
        statistics: Dict[str, Any],
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the relevance of insights to the data.
        
        Args:
            insights_text: Natural language insights
            statistics: Computed statistics
            data_context: Context about the data
            
        Returns:
            Relevance evaluation result
        """
        if not self.client:
            return {"overall_score": 0.5, "feedback": "LLM unavailable"}
        
        prompt = f"""Evaluate how well the following insights match the actual data statistics. Rate on a scale of 0-1 where:
- 1.0 = Insights perfectly align with the data
- 0.7 = Mostly aligned with minor discrepancies
- 0.5 = Some alignment but significant gaps
- 0.3 = Poor alignment, many discrepancies
- 0.0 = Insights do not match the data

Data Statistics:
{json.dumps(statistics, indent=2, default=str)[:1000]}

Insights:
{insights_text}

Check:
1. Do the numeric claims match the statistics?
2. Are the patterns described actually present in the data?
3. Are the insights relevant to the actual data structure?

Provide:
1. An overall relevance score (0-1)
2. Specific alignment checks
3. Any discrepancies found

Respond in JSON format:
{{
    "overall_score": <float 0-1>,
    "numeric_alignment": <float 0-1>,
    "pattern_alignment": <float 0-1>,
    "discrepancies": ["discrepancy1", "discrepancy2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at validating data insights against actual statistics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                eval_result = json.loads(content)
                return {
                    "overall_score": float(eval_result.get("overall_score", 0.5)),
                    "feedback": content,
                    "numeric_alignment": float(eval_result.get("numeric_alignment", 0.5)),
                    "pattern_alignment": float(eval_result.get("pattern_alignment", 0.5)),
                    "discrepancies": eval_result.get("discrepancies", [])
                }
            except json.JSONDecodeError:
                return self._extract_score_from_text(content)
                
        except Exception as e:
            print(f"Warning: LLM relevance evaluation failed: {e}")
            return {
                "overall_score": 0.5,
                "feedback": f"Evaluation failed: {str(e)}"
            }
    
    def _extract_score_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract score from LLM text response if JSON parsing fails.
        
        Args:
            text: LLM response text
            
        Returns:
            Evaluation result with extracted score
        """
        import re
        # Try to find a score in the text
        score_pattern = r'score[:\s]*([0-9.]+)'
        match = re.search(score_pattern, text.lower())
        
        if match:
            try:
                score = float(match.group(1))
                # Normalize to 0-1 if it's in 0-100 range
                if score > 1:
                    score = score / 100
                return {
                    "overall_score": score,
                    "feedback": text
                }
            except ValueError:
                pass
        
        return {
            "overall_score": 0.5,
            "feedback": text
        }

