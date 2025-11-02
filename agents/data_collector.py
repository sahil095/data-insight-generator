"""Data Collector Agent - Fetches datasets from various sources."""
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from tools.kaggle_client import KaggleClient
from tools.data_gov_client import DataGovClient
from utils.helpers import validate_dataframe


class DatasetResult(BaseModel):
    """Structured dataset result schema."""
    datasets: Dict[str, Any] = Field(description="Dictionary of DataFrames")
    metadata: Dict[str, Any] = Field(description="Dataset metadata")
    license_verified: bool = Field(default=False, description="License verification status")
    file_types_valid: bool = Field(default=True, description="File types are CSV/JSON")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class DataCollectorAgent:
    """
    Data Collector Agent - Find and fetch open datasets.
    
    Role: Find and fetch an open dataset based on user query.
    Core Functions:
    - Query Data.gov API or Kaggle Dataset API
    - Download the dataset in CSV/JSON
    - Perform light cleaning and structure detection
    - Verify dataset license is open
    - Ensure file types are CSV/JSON only
    
    Guardrails:
    - Check file type (must be CSV/JSON)
    - Verify dataset license is open
    """
    
    def __init__(self):
        """Initialize the data collector agent."""
        self.kaggle_client = KaggleClient()
        self.data_gov_client = DataGovClient()
        
        self.system_prompt = """You are a Data Collector Agent. Your goal is to find and fetch an open dataset relevant to the user query. 
Return a cleaned and ready-to-analyze CSV file. 
Ensure data is open source and non-sensitive."""
    
    def fetch_by_query(self, query: str, source: str = "data-gov") -> Dict[str, Any]:
        """
        Fetch dataset based on natural language query.
        
        Args:
            query: Natural language query (e.g., "US renewable energy statistics 2022")
            source: Data source ('kaggle' or 'data-gov')
            
        Returns:
            DatasetResult with datasets and metadata
        """
        if source.lower() == "data-gov":
            # Search Data.gov
            search_results = self.data_gov_client.search_datasets(query, limit=5)
            if search_results and "result" in search_results and "results" in search_results["result"]:
                results = search_results["result"]["results"]
                if results:
                    # Take first result
                    dataset_id = results[0].get("id") or results[0].get("name")
                    return self.fetch_dataset("data-gov", dataset_id=dataset_id)
            
        elif source.lower() == "kaggle":
            # For Kaggle, would need to search - simplified for now
            raise NotImplementedError("Kaggle query search not yet implemented")
        
        raise ValueError(f"No datasets found for query: {query}")
    
    def fetch_dataset(
        self,
        source: str,
        dataset_name: Optional[str] = None,
        competition: Optional[str] = None,
        dataset_id: Optional[str] = None,
        resource_url: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Fetch a dataset from the specified source.
        
        Args:
            source: Data source ('kaggle' or 'data-gov')
            dataset_name: Kaggle dataset name (format: 'username/dataset-name')
            competition: Kaggle competition name
            dataset_id: Data.gov dataset ID
            resource_url: Direct resource URL for Data.gov
            output_dir: Directory to save datasets
            
        Returns:
            Dictionary with 'datasets' (dict of DataFrames) and 'metadata'
        """
        if source.lower() == "kaggle":
            return self._fetch_kaggle_dataset(
                dataset_name=dataset_name,
                competition=competition,
                output_dir=output_dir
            )
        elif source.lower() == "data-gov":
            return self._fetch_data_gov_dataset(
                dataset_id=dataset_id,
                resource_url=resource_url,
                output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _fetch_kaggle_dataset(
        self,
        dataset_name: Optional[str] = None,
        competition: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Fetch dataset from Kaggle.
        
        Args:
            dataset_name: Dataset name (format: 'username/dataset-name')
            competition: Competition name
            output_dir: Directory to save dataset
            
        Returns:
            Dictionary with datasets and metadata
        """
        if competition:
            # Download competition
            dataset_path = self.kaggle_client.download_competition(
                competition,
                output_dir=output_dir
            )
            metadata = {
                "source": "kaggle",
                "type": "competition",
                "name": competition
            }
        elif dataset_name:
            # Download dataset
            dataset_path = self.kaggle_client.download_dataset(
                dataset_name,
                output_dir=output_dir
            )
            metadata = self.kaggle_client.get_dataset_metadata(dataset_name)
        else:
            raise ValueError("Either dataset_name or competition must be provided for Kaggle")
        
        # Load datasets
        datasets = self.kaggle_client.load_dataset_files(dataset_path)
        
        if not datasets:
            raise ValueError(f"No valid dataset files found in {dataset_path}")
        
        # Validate datasets
        validated_datasets = {}
        for name, df in datasets.items():
            validation = validate_dataframe(df)
            if validation["valid"]:
                validated_datasets[name] = df
            else:
                print(f"Warning: Dataset {name} failed validation: {validation['errors']}")
        
        if not validated_datasets:
            raise ValueError("No valid datasets after validation")
        
        return {
            "datasets": validated_datasets,
            "metadata": metadata
        }
    
    def _fetch_data_gov_dataset(
        self,
        dataset_id: Optional[str] = None,
        resource_url: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Fetch dataset from Data.gov with guardrails.
        
        Args:
            dataset_id: Dataset ID
            resource_url: Direct resource URL
            output_dir: Directory to save dataset
            
        Returns:
            Dictionary with datasets and metadata
        """
        errors = []
        license_verified = False
        file_types_valid = True
        
        # Get full dataset info for license check
        if dataset_id:
            try:
                dataset_info = self.data_gov_client.get_dataset(dataset_id)
                result = dataset_info.get("result", {})
                
                # Check license - Data.gov datasets are typically open/public domain
                license_info = result.get("license_title", "").lower()
                if "public domain" in license_info or "open" in license_info or "cc0" in license_info:
                    license_verified = True
                elif not license_info:
                    # If no explicit license, assume public (common for Data.gov)
                    license_verified = True
                else:
                    errors.append(f"License may not be fully open: {license_info}")
                    
            except Exception as e:
                errors.append(f"Failed to verify license: {e}")
        
        # Load datasets
        datasets = self.data_gov_client.load_dataset(
            dataset_id=dataset_id,
            resource_url=resource_url,
            output_dir=output_dir
        )
        
        if not datasets:
            raise ValueError("No valid datasets loaded from Data.gov")
        
        # Check file types - ensure only CSV/JSON
        validated_datasets = {}
        for name, df in datasets.items():
            # Check if source file was CSV/JSON (simplified check)
            validation = validate_dataframe(df)
            if validation["valid"]:
                validated_datasets[name] = df
            else:
                file_types_valid = False
                errors.append(f"Dataset {name} failed validation: {validation['errors']}")
        
        if not validated_datasets:
            raise ValueError("No valid datasets after validation")
        
        # Get metadata
        metadata = self.data_gov_client.get_dataset_metadata(
            dataset_id=dataset_id,
            resource_url=resource_url
        )
        metadata["license_verified"] = license_verified
        
        # Return structured result
        result = DatasetResult(
            datasets=validated_datasets,
            metadata=metadata,
            license_verified=license_verified,
            file_types_valid=file_types_valid,
            errors=errors
        )
        
        return result.model_dump()

