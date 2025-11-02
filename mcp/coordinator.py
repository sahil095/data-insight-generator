"""MCP server for coordinating agents using Autogen."""
from typing import Dict, Any, List, Optional
try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("Warning: Autogen not available. MCP coordinator will have limited functionality.")

from config.settings import settings


class MCPServer:
    """
    MCP (Model Context Protocol) server for coordinating multiple agents.
    
    This uses Autogen's multi-agent conversation framework to coordinate
    the data collector, analyst, and auditor agents.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        if not AUTOGEN_AVAILABLE:
            self.agents = {}
            self.group_chat = None
            self.manager = None
            print("Warning: Autogen not installed. Install with: pip install autogen")
            return
        
        # Initialize agents if Autogen is available
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize Autogen agents."""
        if not AUTOGEN_AVAILABLE or not settings.openai_api_key:
            return
        
        # Create system messages for each agent type
        collector_system_message = """You are a data collection agent. Your role is to:
1. Fetch datasets from Kaggle or Data.gov
2. Validate data format and structure
3. Report any issues with data quality
"""
        
        analyst_system_message = """You are a data analyst agent. Your role is to:
1. Perform statistical analysis on datasets
2. Generate visualizations
3. Create natural language insights
4. Ensure statistical accuracy
"""
        
        auditor_system_message = """You are an auditor agent. Your role is to:
1. Validate statistical claims against computed values
2. Check numerical accuracy
3. Ensure consistency in insights
4. Report validation results
"""
        
        # Create agents
        try:
            self.collector_agent = ConversableAgent(
                name="data_collector",
                system_message=collector_system_message,
                llm_config={
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                    "temperature": settings.llm_temperature
                }
            )
            
            self.analyst_agent = ConversableAgent(
                name="data_analyst",
                system_message=analyst_system_message,
                llm_config={
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                    "temperature": settings.llm_temperature
                }
            )
            
            self.auditor_agent = ConversableAgent(
                name="data_auditor",
                system_message=auditor_system_message,
                llm_config={
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                    "temperature": settings.llm_temperature
                }
            )
            
            # Create group chat
            self.agents_list = [
                self.collector_agent,
                self.analyst_agent,
                self.auditor_agent
            ]
            
            self.group_chat = GroupChat(
                agents=self.agents_list,
                messages=[],
                max_round=10
            )
            
            self.manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config={
                    "model": settings.llm_model,
                    "api_key": settings.openai_api_key,
                    "temperature": settings.llm_temperature
                }
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize Autogen agents: {e}")
            self.agents_list = []
            self.group_chat = None
            self.manager = None
    
    def coordinate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate agents to complete a task.
        
        Args:
            task: Description of the task
            context: Additional context for the task
            
        Returns:
            Result dictionary with coordination outcome
        """
        if not AUTOGEN_AVAILABLE or not self.manager:
            return {
                "success": False,
                "error": "Autogen not available or not properly initialized",
                "result": None
            }
        
        try:
            # Prepare initial message
            initial_message = task
            if context:
                initial_message += f"\n\nContext: {context}"
            
            # Start group chat
            result = self.manager.initiate_chat(
                recipient=self.analyst_agent,
                message=initial_message
            )
            
            return {
                "success": True,
                "messages": result.chat_history,
                "result": result.summary if hasattr(result, 'summary') else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def is_available(self) -> bool:
        """
        Check if MCP server is available and properly configured.
        
        Returns:
            True if MCP server is available
        """
        return AUTOGEN_AVAILABLE and self.manager is not None

