"""LangGraph MCP Coordinator - Graph-based agent orchestration."""
from typing import Dict, Any, TypedDict

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: pip install langgraph")

from agents.data_collector import DataCollectorAgent
from agents.analyst import AnalystAgent
from agents.auditor import AuditorAgent
from agents.evaluator import EvaluatorAgent
from config.settings import settings


class AgentState(TypedDict):
    """Shared state for agent workflow."""
    user_query: str
    data_collector_output: Dict[str, Any]
    analyst_output: Dict[str, Any]
    auditor_output: Dict[str, Any]
    evaluator_output: Dict[str, Any]
    current_step: str
    errors: list


class LangGraphCoordinator:
    """
    LangGraph MCP Coordinator for agent orchestration.
    
    Workflow: User → DataCollector → Analyst → Auditor → Evaluator
    Each agent runs as a graph node with shared context via state.
    """
    
    def __init__(self):
        """Initialize the LangGraph coordinator."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")
        
        self.collector = DataCollectorAgent()
        self.analyst = AnalystAgent()
        self.auditor = AuditorAgent()
        self.evaluator = EvaluatorAgent()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("data_collector", self._data_collector_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("auditor", self._auditor_node)
        workflow.add_node("evaluator", self._evaluator_node)
        
        # Define edges
        workflow.set_entry_point("data_collector")
        workflow.add_edge("data_collector", "analyst")
        workflow.add_edge("analyst", "auditor")
        workflow.add_edge("auditor", "evaluator")
        workflow.add_edge("evaluator", END)
        
        return workflow.compile()
    
    def _data_collector_node(self, state: AgentState) -> Dict[str, Any]:
        """Data Collector node."""
        try:
            query = state.get("user_query", "")
            if query:
                # Use query-based fetch
                result = self.collector.fetch_by_query(query, source="data-gov")
            else:
                # Fallback to direct fetch (for backward compatibility)
                result = state.get("data_collector_output", {})
            
            return {
                "data_collector_output": result,
                "current_step": "data_collector",
                "errors": []
            }
        except Exception as e:
            return {
                "data_collector_output": {},
                "current_step": "data_collector",
                "errors": [f"Data collection failed: {str(e)}"]
            }
    
    def _analyst_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyst node."""
        try:
            collector_output = state.get("data_collector_output", {})
            datasets = collector_output.get("datasets", {})
            
            if not datasets:
                raise ValueError("No datasets from collector")
            
            insights = self.analyst.analyze(datasets, generate_visualizations=True)
            
            return {
                "analyst_output": insights,
                "current_step": "analyst",
                "errors": []
            }
        except Exception as e:
            return {
                "analyst_output": {},
                "current_step": "analyst",
                "errors": state.get("errors", []) + [f"Analysis failed: {str(e)}"]
            }
    
    def _auditor_node(self, state: AgentState) -> Dict[str, Any]:
        """Auditor node."""
        try:
            collector_output = state.get("data_collector_output", {})
            analyst_output = state.get("analyst_output", {})
            datasets = collector_output.get("datasets", {})
            
            if not datasets or not analyst_output:
                raise ValueError("Missing inputs for auditor")
            
            validation = self.auditor.validate(analyst_output, datasets)
            
            return {
                "auditor_output": validation,
                "current_step": "auditor",
                "errors": []
            }
        except Exception as e:
            return {
                "auditor_output": {},
                "current_step": "auditor",
                "errors": state.get("errors", []) + [f"Audit failed: {str(e)}"]
            }
    
    def _evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluator node."""
        try:
            analyst_output = state.get("analyst_output", {})
            auditor_output = state.get("auditor_output", {})
            collector_output = state.get("data_collector_output", {})
            
            if not analyst_output or not auditor_output:
                raise ValueError("Missing inputs for evaluator")
            
            # Prepare metadata with dataframes
            metadata = {}
            datasets = collector_output.get("datasets", {})
            for name, df in datasets.items():
                metadata[name] = {
                    "dataframe": df,
                    "shape": df.shape,
                    "columns": list(df.columns)
                }
            
            evaluation = self.evaluator.evaluate(
                analyst_output,
                auditor_output,
                metadata
            )
            
            return {
                "evaluator_output": evaluation,
                "current_step": "evaluator",
                "errors": []
            }
        except Exception as e:
            return {
                "evaluator_output": {},
                "current_step": "evaluator",
                "errors": state.get("errors", []) + [f"Evaluation failed: {str(e)}"]
            }
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the complete workflow.
        
        Args:
            user_query: User's natural language query
            
        Returns:
            Complete workflow results
        """
        initial_state: AgentState = {
            "user_query": user_query,
            "data_collector_output": {},
            "analyst_output": {},
            "auditor_output": {},
            "evaluator_output": {},
            "current_step": "",
            "errors": []
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def run_with_memory(self, user_query: str, memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run workflow with memory persistence.
        
        Args:
            user_query: User's natural language query
            memory: Optional previous state/memory
            
        Returns:
            Complete workflow results with updated memory
        """
        if memory:
            initial_state = memory.copy()
            initial_state["user_query"] = user_query
        else:
            initial_state: AgentState = {
                "user_query": user_query,
                "data_collector_output": {},
                "analyst_output": {},
                "auditor_output": {},
                "evaluator_output": {},
                "current_step": "",
                "errors": []
            }
        
        final_state = self.graph.invoke(initial_state)
        return final_state

