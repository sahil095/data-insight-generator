"""Data.gov API client for fetching datasets."""
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from config.settings import settings


class DataGovClient:
    """Client for interacting with Data.gov API."""
    
    def __init__(self):
        """Initialize Data.gov client."""
        self.api_key = settings.data_gov_api_key
        self.base_url = "https://catalog.data.gov/api/3"
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
    
    def search_datasets(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for datasets on Data.gov.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.base_url}/action/package_search"
        params = {"q": query, "rows": limit}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to search Data.gov: {e}")
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: Dataset ID or package name
            
        Returns:
            Dictionary with dataset information
        """
        url = f"{self.base_url}/action/package_show"
        params = {"id": dataset_id}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to get dataset {dataset_id}: {e}")
    
    def download_resource(
        self,
        resource_url: str,
        output_path: Optional[Path] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Download a resource file from Data.gov.
        
        Args:
            resource_url: URL of the resource to download
            output_path: Directory to save file (default: settings.data_dir)
            filename: Filename to save as (default: from URL)
            
        Returns:
            Path to downloaded file
        """
        if output_path is None:
            output_path = settings.data_dir / "data_gov"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = resource_url.split("/")[-1]
            if "?" in filename:
                filename = filename.split("?")[0]
        
        file_path = output_path / filename
        
        try:
            response = self.session.get(resource_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return file_path
            
        except Exception as e:
            raise ValueError(f"Failed to download resource from {resource_url}: {e}")
    
    def load_dataset(
        self,
        dataset_id: Optional[str] = None,
        resource_url: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a dataset from Data.gov.
        
        Args:
            dataset_id: Dataset ID (optional if resource_url provided)
            resource_url: Direct resource URL (optional if dataset_id provided)
            output_dir: Directory to save files (default: settings.data_dir)
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        datasets = {}
        
        if resource_url:
            # Direct resource URL
            file_path = self.download_resource(resource_url, output_dir)
            df = self._load_file(file_path)
            if df is not None:
                datasets[file_path.stem] = df
        elif dataset_id:
            # Get dataset info and download resources
            dataset_info = self.get_dataset(dataset_id)
            resources = dataset_info.get("result", {}).get("resources", [])
            
            if output_dir is None:
                output_dir = settings.data_dir / "data_gov" / dataset_id
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for resource in resources:
                resource_url = resource.get("url")
                if not resource_url:
                    continue
                
                try:
                    file_path = self.download_resource(resource_url, output_dir)
                    df = self._load_file(file_path)
                    if df is not None:
                        datasets[file_path.stem] = df
                except Exception as e:
                    print(f"Warning: Failed to load resource {resource_url}: {e}")
                    continue
        else:
            raise ValueError("Either dataset_id or resource_url must be provided")
        
        return datasets
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a file into a DataFrame.
        
        Args:
            file_path: Path to file
            
        Returns:
            DataFrame or None if loading fails
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            else:
                print(f"Warning: Unsupported file format: {suffix}")
                return None
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None
    
    def get_dataset_metadata(
        self,
        dataset_id: Optional[str] = None,
        resource_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for a dataset.
        
        Args:
            dataset_id: Dataset ID (optional if resource_url provided)
            resource_url: Direct resource URL (optional if dataset_id provided)
            
        Returns:
            Dictionary with dataset metadata
        """
        if dataset_id:
            dataset_info = self.get_dataset(dataset_id)
            result = dataset_info.get("result", {})
            return {
                "source": "data-gov",
                "id": dataset_id,
                "title": result.get("title", ""),
                "description": result.get("notes", ""),
                "organization": result.get("organization", {}).get("title", ""),
                "tags": [tag.get("name", "") for tag in result.get("tags", [])],
            }
        elif resource_url:
            return {
                "source": "data-gov",
                "resource_url": resource_url,
            }
        else:
            raise ValueError("Either dataset_id or resource_url must be provided")

