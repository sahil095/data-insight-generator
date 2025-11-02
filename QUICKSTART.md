# 🚀 Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) Kaggle account and API credentials
- (Required) OpenAI API key for LLM features

## Step 1: Clone the Repository

```bash
git clone https://github.com/sahil095/data-insight-generator.git
cd data-insight-generator
```

## Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it installs packages like PyTorch, transformers, etc.

## Step 4: Configure API Keys

Create a `.env` file in the project root directory:

```bash
# Windows
notepad .env

# Linux/Mac
nano .env
```

Add your API keys (Groq API key is required by default, or OpenAI for alternative):

```env
# Required for LLM features (default: Groq with Llama models)
GROQ_API_KEY=your_groq_api_key_here

# Optional - Use OpenAI instead (set LLM_PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for Kaggle datasets
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Optional - for Data.gov (usually not needed)
DATA_GOV_API_KEY=your_data_gov_key

# Optional - LLM provider and model configuration
LLM_PROVIDER=groq  # or 'openai'
LLM_MODEL=llama-3.3-70b-versatile  # Groq: llama-3.3-70b-versatile, llama-3.1-70b-versatile | OpenAI: gpt-4, gpt-3.5-turbo
LLM_TEMPERATURE=0.7
```

### Getting API Keys:

1. **Groq API Key (Default - Recommended):**
   - Go to https://console.groq.com/keys
   - Sign up/login (free account available)
   - Create a new API key
   - Copy and paste into `.env` as `GROQ_API_KEY`
   - Provides fast access to Llama models

2. **OpenAI API Key (Alternative):**
   - Go to https://platform.openai.com/api-keys
   - Sign up/login
   - Create a new API key
   - Copy and paste into `.env` as `OPENAI_API_KEY`
   - Set `LLM_PROVIDER=openai` in `.env` to use OpenAI

3. **Kaggle API Key (Optional):**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json`
   - Copy username and key to `.env`

## Step 5: Run the Project

### Option 1: Query-Based Search (Recommended)

Search and analyze datasets using natural language:

```bash
python main.py --query "US renewable energy statistics 2022"
```

Example queries:
```bash
python main.py --query "US renewable energy consumption from 2020 onward"
python main.py --query "COVID-19 vaccination data by state"
python main.py --query "economic indicators unemployment rate"
```

### Option 2: Direct Data.gov Access

If you know the dataset ID:

```bash
python main.py --dataset data-gov --dataset-id "dataset-id-here"
```

### Option 3: Direct Resource URL

```bash
python main.py --dataset data-gov --resource-url "https://data.gov/resource/..."
```

### Option 4: Kaggle Dataset

```bash
python main.py --dataset kaggle --dataset-name "username/dataset-name"
```

Example:
```bash
python main.py --dataset kaggle --dataset-name "shivamb/netflix-shows"
```

### Additional Options

Skip visualizations:
```bash
python main.py --query "your query" --no-viz
```

Use legacy workflow (without LangGraph):
```bash
python main.py --query "your query" --no-langgraph
```

Custom output directory:
```bash
python main.py --query "your query" --output-dir "./my_results"
```

## Step 6: View Results

After running, check the `output/` directory (or your custom output directory):

```
output/
├── insights.json              # Analyst's insights and statistics
├── validation_report.txt      # Auditor's validation with confidence scores
├── evaluation_results.json    # Evaluator's structured scores
├── evaluation_report.txt      # Human-readable evaluation report
└── visualizations/            # Generated charts and plots
    ├── correlation_heatmap.png
    ├── distribution_*.png
    └── ...
```

## Example Output

When you run the project, you'll see:

```
================================================================================
Open Data Insight Generator
================================================================================

Using LangGraph MCP orchestration...

[1/6] Initializing agents...
✓ Agents initialized

[2/6] Collecting dataset...
✓ Collected 1 dataset(s)
  Source: data-gov
  License: Verified (Open)

[3/6] Validating data quality...
✓ Quality check for dataset_name: score 0.85

[4/6] Analyzing datasets...
✓ Analysis complete
  Insights saved to: output/insights.json

[5/6] Auditing insights...
✓ Audit complete
  Report saved to: output/validation_report.txt
  dataset_name: ✓ VALID (confidence: 0.95)

[6/6] Evaluating insight quality...
✓ Evaluation complete
  Results saved to: output/evaluation_results.json
  Report saved to: output/evaluation_report.txt

Evaluation Summary:
  dataset_name:
    Accuracy: 9.2/10
    Clarity: 8.5/10
    Soundness: 9.0/10
    Honesty: 9.5/10
    Numeric Accuracy: 98.5%
```

## Troubleshooting

### Error: "GROQ_API_KEY is required" or "OPENAI_API_KEY is required"
- Make sure your `.env` file exists and contains the appropriate API key
- Default provider is Groq - add `GROQ_API_KEY=your_key`
- To use OpenAI instead, set `LLM_PROVIDER=openai` and add `OPENAI_API_KEY=your_key`
- Check that the `.env` file is in the project root directory

### Error: "LangGraph not available"
- Install LangGraph: `pip install langgraph`
- Or use `--no-langgraph` flag to use legacy workflow

### Error: "Kaggle credentials not set"
- This is a warning, not an error
- You can still use Data.gov datasets without Kaggle credentials
- For Kaggle datasets, add credentials to `.env`

### Error: "No datasets found for query"
- Try a different, more specific query
- Check Data.gov website to see what datasets are available
- Try using `--dataset-id` directly instead of query

### Import Errors
- Make sure you're in the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

### Memory Issues
- Some large datasets may require significant memory
- Try smaller datasets first
- Skip visualizations with `--no-viz` to save memory

## Next Steps

- Explore different datasets and queries
- Modify `config/settings.py` to adjust model settings
- Customize agent prompts in `guardrails/templates.py`
- Add your own evaluation criteria

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review example usage in the code comments
- Check GitHub Issues for common problems

