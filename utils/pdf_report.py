"""PDF Report Generator for analytics, visualizations, and evaluation scores."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.settings import settings

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. Install with: pip install reportlab")


class PDFReportGenerator:
    """Generate PDF reports with analytics, visualizations, and evaluation scores."""
    
    def __init__(self):
        """Initialize PDF report generator."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required. Install with: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#4a7bc8'),
            spaceAfter=8,
            spaceBefore=12
        ))
    
    def generate_report(
        self,
        insights: Dict[str, Any],
        evaluation_results: Optional[Dict[str, Any]] = None,
        validation_report: Optional[str] = None,
        output_path: Optional[Path] = None,
        query: Optional[str] = None
    ) -> Path:
        """
        Generate a comprehensive PDF report.
        
        Args:
            insights: Dictionary of insights from analyst
            evaluation_results: Optional evaluation results
            validation_report: Optional validation report text
            output_path: Path to save PDF
            query: Optional user query
            
        Returns:
            Path to generated PDF
        """
        if output_path is None:
            output_path = settings.output_dir / "analysis_report.pdf"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title page
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Data Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        if query:
            story.append(Paragraph(f"<b>Query:</b> {query}", self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        story.append(PageBreak())
        
        # Process each dataset
        for dataset_name, dataset_insights in insights.items():
            story.extend(self._generate_dataset_section(dataset_name, dataset_insights))
            story.append(PageBreak())
        
        # Evaluation section
        if evaluation_results:
            story.extend(self._generate_evaluation_section(evaluation_results))
            story.append(PageBreak())
        
        # Validation section
        if validation_report:
            story.extend(self._generate_validation_section(validation_report))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _generate_dataset_section(self, dataset_name: str, insights: Dict[str, Any]) -> List:
        """Generate section for a single dataset."""
        story = []
        
        # Dataset header
        story.append(Paragraph(f"Dataset: {dataset_name}", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # EDA Section
        stats = insights.get('statistics', {})
        story.append(Paragraph("Exploratory Data Analysis (EDA)", self.styles['SubSectionHeader']))
        
        # Basic statistics table
        eda_data = [
            ['Metric', 'Value'],
            ['Rows', f"{stats.get('row_count', 'N/A'):,}"],
            ['Columns', f"{stats.get('column_count', 'N/A'):,}"],
            ['Total Cells', f"{stats.get('total_cells', 'N/A'):,}"],
            ['Memory Usage', f"{stats.get('memory_usage_bytes', 0) / 1024 / 1024:.2f} MB"],
            ['Total Nulls', f"{stats.get('total_nulls', 0):,}"],
            ['Null Percentage', f"{stats.get('null_percentage_total', 0):.2f}%"],
            ['Duplicate Rows', f"{stats.get('duplicate_rows', 0):,}"],
            ['Duplicate Percentage', f"{stats.get('duplicate_percentage', 0):.2f}%"],
        ]
        
        data_quality = stats.get('data_quality', {})
        if data_quality:
            eda_data.append(['Data Completeness', f"{data_quality.get('completeness', 0):.2f}%"])
            eda_data.append(['Data Uniqueness', f"{data_quality.get('uniqueness', 0):.2f}%"])
            eda_data.append(['Numeric Columns', f"{data_quality.get('numeric_columns', 0)}"])
            eda_data.append(['Categorical Columns', f"{data_quality.get('categorical_columns', 0)}"])
        
        eda_table = Table(eda_data, colWidths=[3*inch, 2*inch])
        eda_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(eda_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Column information
        if stats.get('columns'):
            story.append(Paragraph("Column Information", self.styles['SubSectionHeader']))
            col_data = [['Column', 'Data Type', 'Null Count', 'Null %']]
            
            for col in stats.get('columns', [])[:20]:  # Limit to 20 columns
                dtype = stats.get('dtypes', {}).get(col, 'N/A')
                null_count = stats.get('null_counts', {}).get(col, 0)
                null_pct = stats.get('null_percentages', {}).get(col, 0)
                col_data.append([col, dtype, str(null_count), f"{null_pct:.2f}%"])
            
            if len(stats.get('columns', [])) > 20:
                col_data.append(['...', f"{len(stats.get('columns', [])) - 20} more columns", '', ''])
            
            col_table = Table(col_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
            col_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a7bc8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(col_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Numeric statistics
        numeric_stats = stats.get('numeric_statistics', {})
        if numeric_stats:
            story.append(Paragraph("Numeric Statistics", self.styles['SubSectionHeader']))
            for col, col_stats in list(numeric_stats.items())[:5]:  # Limit to 5 columns
                story.append(Paragraph(f"<b>{col}</b>", self.styles['Normal']))
                num_data = [
                    ['Statistic', 'Value'],
                    ['Mean', f"{col_stats.get('mean', 'N/A'):.2f}" if isinstance(col_stats.get('mean'), (int, float)) else 'N/A'],
                    ['Median', f"{col_stats.get('median', 'N/A'):.2f}" if isinstance(col_stats.get('median'), (int, float)) else 'N/A'],
                    ['Std Dev', f"{col_stats.get('std', 'N/A'):.2f}" if isinstance(col_stats.get('std'), (int, float)) else 'N/A'],
                    ['Min', f"{col_stats.get('min', 'N/A'):.2f}" if isinstance(col_stats.get('min'), (int, float)) else 'N/A'],
                    ['Max', f"{col_stats.get('max', 'N/A'):.2f}" if isinstance(col_stats.get('max'), (int, float)) else 'N/A'],
                ]
                num_table = Table(num_data, colWidths=[1.5*inch, 1.5*inch])
                num_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6b9bd1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                story.append(num_table)
                story.append(Spacer(1, 0.2*inch))
        
        # Visualizations
        visualizations = insights.get('visualizations', {})
        if visualizations:
            story.append(Paragraph("Visualizations", self.styles['SubSectionHeader']))
            for viz_name, viz_path in list(visualizations.items())[:6]:  # Limit to 6 visualizations
                try:
                    if Path(viz_path).exists():
                        img = Image(viz_path, width=5*inch, height=3.75*inch)
                        story.append(img)
                        story.append(Paragraph(f"<i>{viz_name}</i>", self.styles['Normal']))
                        story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    story.append(Paragraph(f"<i>Could not load visualization: {viz_name}</i>", self.styles['Normal']))
        
        # Insights text
        insights_text = insights.get('insights_text', '')
        if insights_text:
            story.append(Paragraph("Key Insights", self.styles['SubSectionHeader']))
            # Split long text into paragraphs
            for para in insights_text.split('\n'):
                if para.strip():
                    story.append(Paragraph(para.strip(), self.styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _generate_evaluation_section(self, evaluation_results: Dict[str, Any]) -> List:
        """Generate evaluation section."""
        story = []
        
        story.append(Paragraph("Evaluation Results", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        for dataset_name, eval_result in evaluation_results.items():
            story.append(Paragraph(f"<b>{dataset_name}</b>", self.styles['SubSectionHeader']))
            
            eval_data = [
                ['Metric', 'Score'],
                ['Accuracy', f"{eval_result.get('accuracy_score', 0):.1f}/10"],
                ['Clarity', f"{eval_result.get('clarity_score', 0):.1f}/10"],
                ['Soundness', f"{eval_result.get('soundness_score', 0):.1f}/10"],
                ['Honesty', f"{eval_result.get('honesty_score', 0):.1f}/10"],
                ['Numeric Accuracy', f"{eval_result.get('numeric_accuracy', 0)*100:.1f}%"],
            ]
            
            eval_table = Table(eval_data, colWidths=[2*inch, 2*inch])
            eval_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(eval_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Justifications
            justification = eval_result.get('justification', {})
            if justification:
                story.append(Paragraph("Justifications", self.styles['Normal']))
                for key, value in justification.items():
                    story.append(Paragraph(f"<b>{key.capitalize()}:</b> {value}", self.styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _generate_validation_section(self, validation_report: str) -> List:
        """Generate validation section."""
        story = []
        
        story.append(Paragraph("Validation Report", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Split report into paragraphs
        for para in validation_report.split('\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        return story

