"""Configuration settings for the Open Data Insight Generator."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Handle Pydantic v1 vs v2
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field
        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError("pydantic is required. Install with: pip install pydantic")


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    kaggle_username: Optional[str] = Field(default=None, description="Kaggle username")
    kaggle_key: Optional[str] = Field(default=None, description="Kaggle API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    data_gov_api_key: Optional[str] = Field(default=None, description="Data.gov API key (optional)")
    
    # Model Configuration
    llm_model: str = Field(default="gpt-4", description="LLM model to use for analysis")
    llm_temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    
    # Paths
    data_dir: Path = Field(default=Path("./data"), description="Directory for downloaded datasets")
    output_dir: Path = Field(default=Path("./output"), description="Directory for generated outputs")
    
    # Validation thresholds
    min_data_quality_score: float = Field(default=0.5, description="Minimum data quality score")
    numeric_tolerance: float = Field(default=0.01, description="Tolerance for numeric validation")
    
    # Visualization settings
    visualization_format: str = Field(default="png", description="Format for saved visualizations")
    dpi: int = Field(default=300, description="DPI for saved visualizations")
    
    # Pydantic configuration
    if PYDANTIC_V2:
        model_config = {
            "env_file": ".env",
            "case_sensitive": False,
            "env_file_encoding": "utf-8",
            "env_prefix": "",  # No prefix needed
        }
    else:
        class Config:
            """Pydantic v1 config."""
            env_file = ".env"
            case_sensitive = False
            # For v1, env variables are automatically mapped from field names
            fields = {
                'kaggle_username': {'env': 'KAGGLE_USERNAME'},
                'kaggle_key': {'env': 'KAGGLE_KEY'},
                'openai_api_key': {'env': 'OPENAI_API_KEY'},
                'data_gov_api_key': {'env': 'DATA_GOV_API_KEY'},
                'llm_model': {'env': 'LLM_MODEL'},
                'llm_temperature': {'env': 'LLM_TEMPERATURE'},
            }
        
    def validate(self) -> None:
        """Validate required settings."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        
        # Kaggle credentials are optional but recommended for Kaggle datasets
        if not self.kaggle_username or not self.kaggle_key:
            print("⚠ Warning: Kaggle credentials not set. Kaggle dataset fetching will fail.")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance - load from environment variables
settings = Settings()
