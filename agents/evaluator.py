"""Evaluator Agent - LLM-as-a-Judge for qualitative and quantitative evaluation."""
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from evaluation.llm_judge import LLMJudge
from evaluation.numeric_validator import NumericValidator
from config.settings import settings


class EvaluationResult(BaseModel):
    """Structured evaluation result schema."""
    accuracy_score: float = Field(ge=0, le=10, description="Accuracy rating 0-10")
    clarity_score: float = Field(ge=0, le=10, description="Clarity rating 0-10")
    soundness_score: float = Field(ge=0, le=10, description="Statistical soundness rating 0-10")
    honesty_score: float = Field(ge=0, le=10, description="Honesty (no unsupported claims) 0-10")
    numeric_accuracy: float = Field(ge=0, le=1, description="Numeric accuracy percentage 0-1")
    justification: Dict[str, str] = Field(description="Justification for each score")
    discrepancies: list = Field(default_factory=list, description="List of discrepancies found")


class EvaluatorAgent:
    """
    Evaluator Agent - Qualitative and quantitative evaluation using LLM-as-a-Judge.
    
    Role: Evaluate Analyst's report based on:
    - Factual accuracy (cross-check with Auditor findings)
    - Clarity and readability of insights
    - Statistical soundness
    - Honesty (no unsupported claims)
    """
    
    def __init__(self):
        """Initialize the evaluator agent."""
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.llm_judge = LLMJudge()
        self.numeric_validator = NumericValidator()
        
        # System prompt for evaluation
        self.system_prompt = """You are an Evaluation Agent. Evaluate the Analyst's report based on:
1. Factual accuracy (cross-check with Auditor findings)
2. Clarity and readability of insights
3. Statistical soundness
4. Honesty (absence of unsupported claims)

Rate each dimension 0-10 and provide brief justification. Be strict but fair."""
    
    def evaluate(
        self,
        analyst_report: Dict[str, Any],
        auditor_report: Dict[str, Any],
        dataset_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate Analyst's report comprehensively.
        
        Args:
            analyst_report: Analyst's insights and statistics
            auditor_report: Auditor's validation results
            dataset_metadata: Dataset information
            
        Returns:
            Structured evaluation result
        """
        results = {}
        
        for dataset_name, insights in analyst_report.items():
            if dataset_name not in auditor_report:
                continue
            
            # Get auditor validation for this dataset
            auditor_validation = auditor_report[dataset_name]
            
            # Quantitative evaluation (numeric accuracy)
            numeric_eval = self._evaluate_numeric_accuracy(
                insights,
                dataset_name,
                dataset_metadata.get(dataset_name, {})
            )
            
            # Qualitative evaluation (LLM-as-judge)
            qualitative_eval = self._evaluate_qualitative(
                insights,
                auditor_validation,
                dataset_metadata.get(dataset_name, {})
            )
            
            # Combine results
            evaluation_result = {
                "dataset_name": dataset_name,
                "accuracy_score": qualitative_eval.get("accuracy_score", 5.0),
                "clarity_score": qualitative_eval.get("clarity_score", 5.0),
                "soundness_score": qualitative_eval.get("soundness_score", 5.0),
                "honesty_score": qualitative_eval.get("honesty_score", 5.0),
                "numeric_accuracy": numeric_eval.get("accuracy", 0.0),
                "justification": qualitative_eval.get("justification", {}),
                "discrepancies": auditor_validation.get("errors", [])
            }
            
            # Validate with Pydantic schema
            try:
                validated_result = EvaluationResult(**evaluation_result)
                results[dataset_name] = validated_result.dict()
            except Exception as e:
                print(f"Warning: Evaluation result validation failed: {e}")
                results[dataset_name] = evaluation_result
        
        return results
    
    def _evaluate_numeric_accuracy(
        self,
        insights: Dict[str, Any],
        dataset_name: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate numeric accuracy using numeric validator.
        
        Args:
            insights: Analyst's insights
            dataset_name: Name of dataset
            metadata: Dataset metadata
            
        Returns:
            Numeric accuracy evaluation
        """
        if "statistics" not in metadata:
            return {"accuracy": 0.0, "total_claims": 0, "valid_claims": 0}
        
        # Use numeric validator
        numeric_result = self.numeric_validator.validate_insight_numerics(
            insights.get("insights_text", ""),
            {dataset_name: insights.get("statistics", {})},
            {dataset_name: metadata.get("dataframe")} if "dataframe" in metadata else {}
        )
        
        return {
            "accuracy": numeric_result.get("accuracy", 0.0),
            "total_claims": numeric_result.get("total_claims", 0),
            "valid_claims": numeric_result.get("valid_claims", 0)
        }
    
    def _evaluate_qualitative(
        self,
        insights: Dict[str, Any],
        auditor_validation: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate qualitative aspects using LLM-as-judge.
        
        Args:
            insights: Analyst's insights
            auditor_validation: Auditor's validation results
            metadata: Dataset metadata
            
        Returns:
            Qualitative evaluation scores
        """
        if not self.client:
            return {
                "accuracy_score": 5.0,
                "clarity_score": 5.0,
                "soundness_score": 5.0,
                "honesty_score": 5.0,
                "justification": {"note": "LLM evaluation unavailable"}
            }
        
        insights_text = insights.get("insights_text", "")
        statistics = insights.get("statistics", {})
        
        # Prepare evaluation prompt
        prompt = f"""Evaluate the Analyst's report based on the following criteria:

ANALYST'S INSIGHTS:
{insights_text[:2000]}

AUDITOR'S VALIDATION:
- Overall Valid: {auditor_validation.get('overall_valid', False)}
- Errors: {len(auditor_validation.get('errors', []))}
- Errors Details: {', '.join(auditor_validation.get('errors', [])[:5])}

DATASET CONTEXT:
- Shape: {statistics.get('shape', 'unknown')}
- Columns: {len(statistics.get('columns', []))}

EVALUATION DIMENSIONS:
1. Accuracy (0-10): Are insights statistically supported? Cross-check with Auditor findings.
2. Clarity (0-10): Are explanations coherent and useful?
3. Soundness (0-10): Is the statistical analysis methodologically sound?
4. Honesty (0-10): Any unsupported claims or exaggerations?

Respond in JSON format:
{{
    "accuracy_score": <float 0-10>,
    "clarity_score": <float 0-10>,
    "soundness_score": <float 0-10>,
    "honesty_score": <float 0-10>,
    "justification": {{
        "accuracy": "<brief justification>",
        "clarity": "<brief justification>",
        "soundness": "<brief justification>",
        "honesty": "<brief justification>"
    }}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"} if hasattr(self.client.chat.completions.create, "__code__") else None
            )
            
            content = response.choices[0].message.content.strip()
            eval_result = json.loads(content)
            
            return {
                "accuracy_score": float(eval_result.get("accuracy_score", 5.0)),
                "clarity_score": float(eval_result.get("clarity_score", 5.0)),
                "soundness_score": float(eval_result.get("soundness_score", 5.0)),
                "honesty_score": float(eval_result.get("honesty_score", 5.0)),
                "justification": eval_result.get("justification", {})
            }
            
        except Exception as e:
            print(f"Warning: LLM qualitative evaluation failed: {e}")
            return {
                "accuracy_score": 5.0,
                "clarity_score": 5.0,
                "soundness_score": 5.0,
                "honesty_score": 5.0,
                "justification": {"error": str(e)}
            }
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            evaluation_results: Results from evaluate()
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION REPORT (LLM-as-a-Judge)")
        lines.append("=" * 80)
        lines.append("")
        
        for dataset_name, result in evaluation_results.items():
            lines.append(f"Dataset: {dataset_name}")
            lines.append("-" * 80)
            lines.append(f"Accuracy Score: {result.get('accuracy_score', 0):.1f}/10")
            lines.append(f"Clarity Score: {result.get('clarity_score', 0):.1f}/10")
            lines.append(f"Soundness Score: {result.get('soundness_score', 0):.1f}/10")
            lines.append(f"Honesty Score: {result.get('honesty_score', 0):.1f}/10")
            lines.append(f"Numeric Accuracy: {result.get('numeric_accuracy', 0)*100:.1f}%")
            lines.append("")
            
            justification = result.get("justification", {})
            if justification:
                lines.append("Justifications:")
                for key, value in justification.items():
                    lines.append(f"  {key.capitalize()}: {value}")
                lines.append("")
            
            discrepancies = result.get("discrepancies", [])
            if discrepancies:
                lines.append(f"Discrepancies Found: {len(discrepancies)}")
                for disc in discrepancies[:5]:
                    lines.append(f"  - {disc}")
                lines.append("")
            
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)

