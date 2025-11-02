"""Main orchestration script for Open Data Insight Generator."""
import argparse
import json
from pathlib import Path

try:
    from mcp.langgraph_coordinator import LangGraphCoordinator
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.data_collector import DataCollectorAgent
from agents.analyst import AnalystAgent
from agents.auditor import AuditorAgent
from agents.evaluator import EvaluatorAgent
from guardrails.validators import DataQualityGuardrail
from config.settings import settings


def run_langgraph_workflow(query: str, output_dir: Path) -> None:
    """Run workflow using LangGraph coordinator."""
    print("Using LangGraph MCP orchestration...")
    print()
    
    coordinator = LangGraphCoordinator()
    result = coordinator.run(query)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save insights
    if result.get("analyst_output"):
        insights_path = output_dir / "insights.json"
        analyst = AnalystAgent()
        analyst.save_insights(result["analyst_output"], str(insights_path))
        print(f"✓ Insights saved to: {insights_path}")
    
    # Save validation report
    if result.get("auditor_output"):
        auditor = AuditorAgent()
        report = auditor.generate_validation_report(result["auditor_output"])
        report_path = output_dir / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Validation report saved to: {report_path}")
    
    # Save evaluation report
    if result.get("evaluator_output"):
        evaluator = EvaluatorAgent()
        eval_report = evaluator.generate_evaluation_report(result["evaluator_output"])
        eval_path = output_dir / "evaluation_report.txt"
        with open(eval_path, 'w') as f:
            f.write(eval_report)
        
        eval_json_path = output_dir / "evaluation_results.json"
        with open(eval_json_path, 'w') as f:
            json.dump(result["evaluator_output"], f, indent=2, default=str)
        
        print(f"✓ Evaluation report saved to: {eval_path}")
        
        # Print summary scores
        print()
        print("Evaluation Summary:")
        for dataset_name, eval_result in result["evaluator_output"].items():
            print(f"  {dataset_name}:")
            print(f"    Accuracy: {eval_result.get('accuracy_score', 0):.1f}/10")
            print(f"    Clarity: {eval_result.get('clarity_score', 0):.1f}/10")
            print(f"    Soundness: {eval_result.get('soundness_score', 0):.1f}/10")
            print(f"    Honesty: {eval_result.get('honesty_score', 0):.1f}/10")
            print(f"    Numeric Accuracy: {eval_result.get('numeric_accuracy', 0)*100:.1f}%")


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Open Data Insight Generator - Automated dataset analysis with validation"
    )
    
    # Query-based input (new)
    parser.add_argument(
        "--query",
        help="Natural language query (e.g., 'US renewable energy statistics 2022')"
    )
    
    # Legacy arguments
    parser.add_argument(
        "--dataset",
        choices=["kaggle", "data-gov"],
        help="Data source: 'kaggle' or 'data-gov' (use with --dataset-id/--dataset-name)"
    )
    
    parser.add_argument(
        "--dataset-name",
        help="Kaggle dataset name (format: username/dataset-name)"
    )
    
    parser.add_argument(
        "--competition",
        help="Kaggle competition name"
    )
    
    parser.add_argument(
        "--dataset-id",
        help="Data.gov dataset ID"
    )
    
    parser.add_argument(
        "--resource-url",
        help="Direct resource URL for Data.gov"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        default=True,
        help="Use LangGraph MCP orchestration (default: True)"
    )
    
    parser.add_argument(
        "--no-langgraph",
        action="store_false",
        dest="use_langgraph",
        help="Use legacy sequential workflow instead of LangGraph"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Open Data Insight Generator")
    print("=" * 80)
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If query provided, use LangGraph workflow
    if args.query:
        if args.use_langgraph and LANGGRAPH_AVAILABLE:
            run_langgraph_workflow(args.query, output_dir)
        else:
            print("Error: LangGraph required for query-based input")
            print("Install with: pip install langgraph")
            return
        return
    
    # Legacy workflow (sequential)
    print("Using legacy sequential workflow...")
    print()
    
    # Initialize agents
    print("[1/6] Initializing agents...")
    collector = DataCollectorAgent()
    analyst = AnalystAgent()
    auditor = AuditorAgent()
    evaluator = EvaluatorAgent()
    quality_guardrail = DataQualityGuardrail()
    
    print("✓ Agents initialized")
    print()
    
    # Step 1: Collect data
    print("[2/6] Collecting dataset...")
    try:
        if args.dataset == "kaggle":
            if not args.dataset_name and not args.competition:
                print("Error: --dataset-name or --competition required for Kaggle")
                return
            data_result = collector.fetch_dataset(
                "kaggle",
                dataset_name=args.dataset_name,
                competition=args.competition
            )
        elif args.dataset == "data-gov":
            if not args.dataset_id and not args.resource_url:
                print("Error: --dataset-id or --resource-url required for Data.gov")
                return
            data_result = collector.fetch_dataset(
                "data-gov",
                dataset_id=args.dataset_id,
                resource_url=args.resource_url
            )
        else:
            print("Error: --dataset required or use --query for query-based search")
            return
        
        datasets = data_result["datasets"]
        metadata = data_result["metadata"]
        
        print(f"✓ Collected {len(datasets)} dataset(s)")
        print(f"  Source: {metadata.get('source', 'unknown')}")
        if data_result.get("license_verified"):
            print(f"  License: Verified (Open)")
        print()
        
    except Exception as e:
        print(f"✗ Failed to collect dataset: {e}")
        return
    
    # Step 2: Validate data quality
    print("[3/6] Validating data quality...")
    quality_results = {}
    for name, df in datasets.items():
        quality = quality_guardrail.validate_data_quality(df)
        quality_results[name] = quality
        
        if not quality["valid"]:
            print(f"⚠ Data quality issues for {name}:")
            for issue in quality["issues"]:
                print(f"  - {issue}")
        
        print(f"✓ Quality check for {name}: score {quality['score']:.2f}")
    
    print()
    
    # Step 3: Analyze data
    print("[4/6] Analyzing datasets...")
    try:
        insights = analyst.analyze(
            datasets,
            generate_visualizations=not args.no_viz
        )
        
        # Save insights
        insights_path = output_dir / "insights.json"
        analyst.save_insights(insights, str(insights_path))
        
        print(f"✓ Analysis complete")
        print(f"  Insights saved to: {insights_path}")
        print()
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return
    
    # Step 4: Audit insights
    print("[5/6] Auditing insights...")
    try:
        validation_results = auditor.validate(insights, datasets)
        
        # Generate validation report
        report = auditor.generate_validation_report(validation_results)
        report_path = output_dir / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Audit complete")
        print(f"  Report saved to: {report_path}")
        print()
        
        # Print summary with confidence scores
        for dataset_name, result in validation_results.items():
            confidence = result.get("confidence_score", 0.0)
            valid = result.get("overall_valid", False)
            status = "✓ VALID" if valid else "✗ INVALID"
            print(f"  {dataset_name}: {status} (confidence: {confidence:.2f})")
        
        print()
        
    except Exception as e:
        print(f"✗ Auditing failed: {e}")
        return
    
    # Step 5: Evaluation
    print("[6/6] Evaluating insight quality...")
    try:
        # Prepare metadata
        metadata_dict = {}
        for dataset_name, df in datasets.items():
            metadata_dict[dataset_name] = {
                "dataframe": df,
                "shape": df.shape,
                "columns": list(df.columns)
            }
        
        evaluation_results = evaluator.evaluate(
            insights,
            validation_results,
            metadata_dict
        )
        
        # Save evaluation results
        eval_path = output_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate evaluation report
        eval_report = evaluator.generate_evaluation_report(evaluation_results)
        eval_report_path = output_dir / "evaluation_report.txt"
        with open(eval_report_path, 'w') as f:
            f.write(eval_report)
        
        print(f"✓ Evaluation complete")
        print(f"  Results saved to: {eval_path}")
        print(f"  Report saved to: {eval_report_path}")
        print()
        
        # Print summary scores
        print("Evaluation Summary:")
        for dataset_name, result in evaluation_results.items():
            print(f"  {dataset_name}:")
            print(f"    Accuracy: {result.get('accuracy_score', 0):.1f}/10")
            print(f"    Clarity: {result.get('clarity_score', 0):.1f}/10")
            print(f"    Soundness: {result.get('soundness_score', 0):.1f}/10")
            print(f"    Honesty: {result.get('honesty_score', 0):.1f}/10")
            print(f"    Numeric Accuracy: {result.get('numeric_accuracy', 0)*100:.1f}%")
        print()
        
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
        print()
    
    # Final summary
    print("=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - insights.json")
    print(f"  - validation_report.txt")
    print(f"  - evaluation_results.json")
    print(f"  - evaluation_report.txt")
    if not args.no_viz:
        print(f"  - visualizations/")
    print()


if __name__ == "__main__":
    settings.validate()
    main()
