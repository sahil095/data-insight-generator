# Open Data Insight Generator

An end-to-end modular system that automatically fetches open datasets (Data.gov, Kaggle), analyzes them, generates insights and visualizations, verifies statistical accuracy, and evaluates quality using LLM-as-a-judge.

## 🎯 Goal

Automatically fetch an open dataset, analyze it, generate insights and visualizations, verify the results, and evaluate the quality of generated insights using both numeric checks and LLM-as-a-judge.

## 🏗️ System Overview

**High-Level Flow:**
```
User Query → Data Collector Agent → Analyst Agent → Auditor Agent → Evaluator Agent (LLM-as-Judge)
```

Each agent runs as a node connected via **LangGraph MCP** for message passing and shared context.

## 🏗️ Project Structure

```
open-data-insight-generator/
├── agents/
│   ├── __init__.py
│   ├── data_collector.py      # Finds and fetches datasets (with license verification)
│   ├── analyst.py             # Data analysis and visualization
│   ├── auditor.py             # Validates statistical accuracy (with confidence scores)
│   └── evaluator.py           # LLM-as-judge evaluation (unified)
├── tools/
│   ├── __init__.py
│   ├── kaggle_client.py       # Kaggle API wrapper
│   ├── data_gov_client.py     # Data.gov API wrapper
│   └── visualization.py       # Visualization utilities
├── mcp/
│   ├── __init__.py
│   ├── coordinator.py         # Autogen MCP (legacy)
│   └── langgraph_coordinator.py  # LangGraph MCP (primary)
├── guardrails/
│   ├── __init__.py
│   ├── validators.py          # Statistical validation
│   └── templates.py            # Prompt templates with reasoning chains
├── evaluation/
│   ├── __init__.py
│   ├── llm_judge.py           # LLM-as-a-judge evaluation
│   └── numeric_validator.py   # Numeric correctness checks
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Utility functions
├── main.py                    # Main orchestration script
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up API credentials:**
Create a `.env` file in the project root:
```
# Required for LLM features (default: Groq)
GROQ_API_KEY=your_groq_api_key

# Optional - Use OpenAI instead (set LLM_PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key

# Optional - for Kaggle datasets
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Optional - for Data.gov
DATA_GOV_API_KEY=your_data_gov_key

# Optional - LLM configuration
LLM_PROVIDER=groq  # or 'openai'
LLM_MODEL=llama-3.3-70b-versatile  # Groq: llama-3.3-70b-versatile | OpenAI: gpt-4
```

3. **Run with query-based input (recommended):**
```bash
python main.py --query "US renewable energy statistics 2022"
```

4. **Or use direct dataset access:**
```bash
python main.py --dataset data-gov --dataset-id "dataset-id"
python main.py --dataset kaggle --dataset-name "username/dataset-name"
```

## 🤖 Agents Overview

### 1️⃣ Data Collector Agent

**Role:** Find and fetch an open dataset based on user query (e.g., "US renewable energy statistics 2022").

**Core Functions:**
- Query Data.gov API or Kaggle Dataset API
- Download the dataset in CSV/JSON
- Perform light cleaning and structure detection
- **Verify dataset license is open**
- **Ensure file types are CSV/JSON only**

**Guardrails:**
- Check file type (must be CSV/JSON)
- Verify dataset license is open

**Prompt:**
```
You are a Data Collector Agent. Your goal is to find and fetch an open dataset relevant to the user query. 
Return a cleaned and ready-to-analyze CSV file. 
Ensure data is open source and non-sensitive.

User Query: {query}
```

### 2️⃣ Analyst Agent

**Role:** Perform statistical analysis and generate insights.

**Core Functions:**
- Performs statistical analysis (mean, variance, correlations, etc.)
- Generates visualizations (charts, plots, distributions)
- Creates natural language summaries with **structured JSON outputs**
- Uses **tool-augmented reasoning** (pandas before summarization)
- **Self-consistency prompting**: Re-evaluates top insights

### 3️⃣ Auditor Agent

**Role:** Validate analytical correctness and numerical consistency.

**Core Functions:**
- Recalculate basic metrics (means, counts, correlations)
- Compare with Analyst's reported numbers
- Flag discrepancies
- **Return confidence score (0-1)**
- **Generate discrepancy reports**

**Output Schema:**
- `confidence_score`: 0-1 confidence level
- `discrepancies`: List of found issues
- `overall_valid`: Boolean validation status

### 4️⃣ Evaluator Agent (LLM-as-a-Judge)

**Role:** Qualitative and quantitative evaluation of the system.

**Evaluation Dimensions:**
- **Accuracy** (0-10): Are insights statistically supported?
- **Clarity** (0-10): Are explanations coherent and useful?
- **Soundness** (0-10): Is analysis methodologically sound?
- **Honesty** (0-10): Any unsupported claims?
- **Numeric Accuracy**: Percentage of validated numeric claims

**Inputs:**
- Dataset metadata
- Analyst's summary
- Auditor's report

**Prompt:**
```
You are an Evaluation Agent. Evaluate the Analyst's report based on:
1. Factual accuracy (cross-check with Auditor findings)
2. Clarity and readability of insights
3. Statistical soundness
4. Honesty (no unsupported claims)

Rate each 0-10 and justify briefly.
```

## 🧭 MCP Orchestration

**LangGraph MCP** is used for graph-based agent coordination:

- Each agent runs as a **graph node**
- MCP handles **context passing**, **memory persistence**, and **error retries**
- Workflow graph: `User → DataCollector → Analyst → Auditor → Evaluator`

**Shared Context:**
Each agent's output becomes structured input for the next via MCP's message context.

## 🧱 Prompt Engineering Techniques

- **Structured prompting** with JSON schema outputs for clean inter-agent passing
- **Self-consistency prompting**: Analyst re-evaluates top 2-3 insights
- **Tool-augmented reasoning**: Analyst uses pandas before summarization
- **Instruction + role prompting**: Keeps each agent tightly scoped

## 🧪 Evaluation System

### Quantitative Evaluation
- Compare computed vs. reported statistics (numeric accuracy %)
- Cross-check data coverage (rows analyzed vs. total rows)

### Qualitative Evaluation
- **LLM-as-a-Judge** evaluation scores (accuracy, clarity, soundness, honesty)
- Confidence scores from Auditor
- Optional human review mode for demos

## 🚀 POC Demo Flow

**User Input:**
```bash
python main.py --query "Analyze open data about U.S. renewable energy consumption from 2020 onward"
```

**DataCollector Agent:**
- Searches Data.gov for renewable energy datasets
- Fetches CSV, cleans it
- Verifies license is open

**Analyst Agent:**
- Generates summary: "Renewable energy grew 18% between 2020–2023"
- Identifies: "Wind energy is the fastest-growing sector"
- Creates visualization: Year vs. Production chart

**Auditor Agent:**
- Confirms numbers match raw data
- Reports: "All stats verified; minor rounding differences"
- Confidence score: 0.95

**Evaluator Agent (LLM-as-Judge):**
- Rates analysis clarity (9/10), accuracy (9/10), soundness (8/10), honesty (9/10)
- Numeric accuracy: 98%

**Output:**
- Interactive summary dashboard + evaluation report
- All files saved to `./output/`

## 📊 Output Files

- `insights.json` - Analyst's insights and statistics
- `validation_report.txt` - Auditor's validation with confidence scores
- `evaluation_results.json` - Evaluator's structured scores
- `evaluation_report.txt` - Human-readable evaluation report
- `visualizations/` - Generated charts and plots

## 🔧 Configuration

Modify `config/settings.py` to adjust:
- Model selection (OpenAI, Hugging Face, local models)
- Agent behavior parameters
- Validation thresholds
- Evaluation criteria

## 📝 Example Usage

**Query-based (recommended):**
```python
from mcp.langgraph_coordinator import LangGraphCoordinator

coordinator = LangGraphCoordinator()
result = coordinator.run("US renewable energy statistics 2022")
```

**Legacy sequential workflow:**
```python
from agents.data_collector import DataCollectorAgent
from agents.analyst import AnalystAgent
from agents.auditor import AuditorAgent
from agents.evaluator import EvaluatorAgent

# Initialize agents
collector = DataCollectorAgent()
analyst = AnalystAgent()
auditor = AuditorAgent()
evaluator = EvaluatorAgent()

# Run pipeline
data = collector.fetch_dataset("data-gov", dataset_id="...")
insights = analyst.analyze(data["datasets"])
validation = auditor.validate(insights, data["datasets"])
evaluation = evaluator.evaluate(insights, validation, metadata)
```

## 📄 License

MIT License - Open source and free to use.
