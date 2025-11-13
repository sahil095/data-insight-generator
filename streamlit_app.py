"""Streamlit application for Open Data Insight Generator."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional
import io

try:
    from mcp.langgraph_coordinator import LangGraphCoordinator
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.analyst import AnalystAgent
from agents.auditor import AuditorAgent
from agents.evaluator import EvaluatorAgent
from agents.data_collector import DataCollectorAgent
try:
    from utils.pdf_report import PDFReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="Open Data Insight Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'query' not in st.session_state:
    st.session_state.query = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'validation_report' not in st.session_state:
    st.session_state.validation_report = None


def load_csv_from_upload(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Load CSV from uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)
        filename = uploaded_file.name.replace('.csv', '')
        return {filename: df}
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return {}


def run_analysis(datasets: Dict[str, pd.DataFrame], query: Optional[str] = None):
    """Run the complete analysis pipeline."""
    with st.spinner("Running analysis pipeline..."):
        try:
            # Initialize agents
            analyst = AnalystAgent()
            auditor = AuditorAgent()
            evaluator = EvaluatorAgent()
            
            # Step 1: Analyze
            st.info("Analyzing datasets...")
            insights = analyst.analyze(datasets, generate_visualizations=True, query=query)
            
            # Step 2: Audit
            st.info("Validating insights...")
            validation_results = auditor.validate(insights, datasets)
            validation_report = auditor.generate_validation_report(validation_results)
            
            # Step 3: Evaluate
            st.info("Evaluating insight quality...")
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
            
            # Store in session state
            st.session_state.datasets = datasets
            st.session_state.insights = insights
            st.session_state.evaluation_results = evaluation_results
            st.session_state.validation_report = validation_report
            st.session_state.query = query
            
            st.success("Analysis complete!")
            return True
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return False


def display_eda(stats: Dict[str, Any], dataset_name: str):
    """Display EDA results."""
    st.subheader(f"📈 Exploratory Data Analysis: {dataset_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{stats.get('row_count', 0):,}")
    with col2:
        st.metric("Columns", f"{stats.get('column_count', 0):,}")
    with col3:
        st.metric("Total Nulls", f"{stats.get('total_nulls', 0):,}")
    with col4:
        null_pct = stats.get('null_percentage_total', 0)
        st.metric("Null %", f"{null_pct:.2f}%")
    
    # Data Quality Metrics
    data_quality = stats.get('data_quality', {})
    if data_quality:
        st.subheader("Data Quality Metrics")
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            st.metric("Completeness", f"{data_quality.get('completeness', 0):.2f}%")
        with qcol2:
            st.metric("Uniqueness", f"{data_quality.get('uniqueness', 0):.2f}%")
        with qcol3:
            st.metric("Numeric Columns", data_quality.get('numeric_columns', 0))
        with qcol4:
            st.metric("Categorical Columns", data_quality.get('categorical_columns', 0))
    
    # Schema Information
    with st.expander("📋 Schema Information"):
        schema_data = []
        for col in stats.get('columns', []):
            dtype = stats.get('dtypes', {}).get(col, 'N/A')
            null_count = stats.get('null_counts', {}).get(col, 0)
            null_pct = stats.get('null_percentages', {}).get(col, 0)
            schema_data.append({
                'Column': col,
                'Data Type': dtype,
                'Null Count': null_count,
                'Null %': f"{null_pct:.2f}%"
            })
        if schema_data:
            st.dataframe(pd.DataFrame(schema_data), width='stretch')
    
    # Numeric Statistics
    numeric_stats = stats.get('numeric_statistics', {})
    if numeric_stats:
        with st.expander("🔢 Numeric Statistics"):
            for col, col_stats in list(numeric_stats.items())[:5]:
                st.write(f"**{col}**")
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.write(f"Mean: {col_stats.get('mean', 'N/A'):.2f}" if isinstance(col_stats.get('mean'), (int, float)) else "Mean: N/A")
                with stat_cols[1]:
                    st.write(f"Median: {col_stats.get('median', 'N/A'):.2f}" if isinstance(col_stats.get('median'), (int, float)) else "Median: N/A")
                with stat_cols[2]:
                    st.write(f"Std: {col_stats.get('std', 'N/A'):.2f}" if isinstance(col_stats.get('std'), (int, float)) else "Std: N/A")
                with stat_cols[3]:
                    st.write(f"Range: [{col_stats.get('min', 'N/A'):.2f}, {col_stats.get('max', 'N/A'):.2f}]" 
                            if isinstance(col_stats.get('min'), (int, float)) and isinstance(col_stats.get('max'), (int, float)) 
                            else "Range: N/A")
                st.divider()
    
    # Categorical Statistics
    cat_stats = stats.get('categorical_statistics', {})
    if cat_stats:
        with st.expander("📊 Categorical Statistics"):
            for col, col_stats in list(cat_stats.items())[:5]:
                st.write(f"**{col}**")
                st.write(f"Unique values: {col_stats.get('unique_count', 0)}")
                st.write(f"Mode: {col_stats.get('mode', 'N/A')}")
                st.divider()


def display_visualizations(visualizations: Dict[str, str]):
    """Display visualizations."""
    if not visualizations:
        st.info("No visualizations available.")
        return
    
    st.subheader("📊 Visualizations")
    
    # Group visualizations by type
    viz_list = list(visualizations.items())
    for i in range(0, len(viz_list), 2):
        cols = st.columns(2)
        for j, (viz_name, viz_path) in enumerate(viz_list[i:i+2]):
            if j < len(cols):
                with cols[j]:
                    try:
                        if Path(viz_path).exists():
                            st.image(viz_path, caption=viz_name, width='stretch')
                    except Exception as e:
                        st.error(f"Could not load {viz_name}: {e}")


def main():
    """Main Streamlit application."""
    st.title("📊 Open Data Insight Generator")
    st.markdown("Automated dataset analysis with validation and evaluation")
    
    # Sidebar
    with st.sidebar:
        st.header("Input Options")
        
        # CSV Upload
        st.subheader("Option 1: Upload CSV")
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=['csv'],
            help="Upload your CSV file for analysis"
        )
        
        st.divider()
        
        # Query Input
        st.subheader("Option 2: Query Online Data")
        query = st.text_input(
            "Enter your query",
            placeholder="e.g., US renewable energy statistics 2022",
            help="Search for datasets from Data.gov"
        )
        
        use_online = st.checkbox("Use online data source", value=False)
        
        st.divider()
        
        # Run Analysis Button
        if st.button("🚀 Run Analysis", type="primary", width='stretch'):
            datasets = {}
            
            # Priority: CSV upload over online query
            if uploaded_file is not None:
                datasets = load_csv_from_upload(uploaded_file)
                if datasets:
                    run_analysis(datasets, query=None)
            elif use_online and query:
                if not LANGGRAPH_AVAILABLE:
                    st.error("LangGraph is required for online queries. Install with: pip install langgraph")
                else:
                    try:
                        with st.spinner("Fetching data from online source..."):
                            coordinator = LangGraphCoordinator()
                            result = coordinator.run(query)
                            
                            datasets = result.get("data_collector_output", {}).get("datasets", {})
                            if datasets:
                                # Extract insights from result
                                st.session_state.insights = result.get("analyst_output", {})
                                st.session_state.evaluation_results = result.get("evaluator_output", {})
                                
                                # Get validation report
                                if result.get("auditor_output"):
                                    auditor = AuditorAgent()
                                    st.session_state.validation_report = auditor.generate_validation_report(
                                        result["auditor_output"]
                                    )
                                
                                st.session_state.datasets = datasets
                                st.session_state.query = query
                                st.success("Analysis complete!")
                            else:
                                st.error("No datasets found for the query.")
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")
            else:
                st.warning("Please upload a CSV file or enter a query with 'Use online data source' checked.")
        
        # Clear Results
        if st.button("🗑️ Clear Results", width='stretch'):
            st.session_state.results = None
            st.session_state.query = None
            st.session_state.datasets = None
            st.session_state.insights = None
            st.session_state.evaluation_results = None
            st.session_state.validation_report = None
            st.rerun()
    
    # Main Content Area
    if st.session_state.insights:
        # Display results
        st.header("Analysis Results")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Insights", "📈 Visualizations", "✅ Evaluation", "📄 Validation"])
        
        with tab1:
            for dataset_name, insights in st.session_state.insights.items():
                stats = insights.get('statistics', {})
                display_eda(stats, dataset_name)
                
                st.divider()
                
                # Insights Text
                insights_text = insights.get('insights_text', '')
                if insights_text:
                    st.subheader("💡 Key Insights")
                    st.write(insights_text)
        
        with tab2:
            for dataset_name, insights in st.session_state.insights.items():
                visualizations = insights.get('visualizations', {})
                if visualizations:
                    st.subheader(f"Visualizations: {dataset_name}")
                    display_visualizations(visualizations)
        
        with tab3:
            if st.session_state.evaluation_results:
                st.subheader("Evaluation Metrics")
                for dataset_name, eval_result in st.session_state.evaluation_results.items():
                    st.write(f"**{dataset_name}**")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Accuracy", f"{eval_result.get('accuracy_score', 0):.1f}/10")
                    with col2:
                        st.metric("Clarity", f"{eval_result.get('clarity_score', 0):.1f}/10")
                    with col3:
                        st.metric("Soundness", f"{eval_result.get('soundness_score', 0):.1f}/10")
                    with col4:
                        st.metric("Honesty", f"{eval_result.get('honesty_score', 0):.1f}/10")
                    with col5:
                        st.metric("Numeric Accuracy", f"{eval_result.get('numeric_accuracy', 0)*100:.1f}%")
                    
                    # Justifications
                    justification = eval_result.get('justification', {})
                    if justification:
                        with st.expander("View Justifications"):
                            for key, value in justification.items():
                                st.write(f"**{key.capitalize()}:** {value}")
                    
                    st.divider()
        
        with tab4:
            if st.session_state.validation_report:
                st.subheader("Validation Report")
                st.text(st.session_state.validation_report)
        
        # PDF Report Generation
        st.divider()
        st.header("📥 Download Report")
        
        if not PDF_AVAILABLE:
            st.warning("PDF generation requires reportlab. Install with: pip install reportlab")
        elif st.button("📄 Generate PDF Report", type="primary", width='stretch'):
            try:
                with st.spinner("Generating PDF report..."):
                    pdf_generator = PDFReportGenerator()
                    output_path = Path(settings.output_dir) / "analysis_report.pdf"
                    
                    pdf_path = pdf_generator.generate_report(
                        insights=st.session_state.insights,
                        evaluation_results=st.session_state.evaluation_results,
                        validation_report=st.session_state.validation_report,
                        output_path=output_path,
                        query=st.session_state.query
                    )
                    
                    # Read PDF and provide download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name="analysis_report.pdf",
                            mime="application/pdf",
                            width='stretch'
                        )
                    st.success("PDF report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
    
    else:
        # Welcome message
        st.info("👈 Use the sidebar to upload a CSV file or enter a query to get started!")


if __name__ == "__main__":
    main()

