"""Kaggle API client for fetching datasets."""
import os
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import kaggle
from config.settings import settings


class KaggleClient:
    """Client for interacting with Kaggle API."""
    
    def __init__(self):
        """Initialize Kaggle client with credentials."""
        self.username = settings.kaggle_username
        self.key = settings.kaggle_key
        
        if self.username and self.key:
            os.environ["KAGGLE_USERNAME"] = self.username
            os.environ["KAGGLE_KEY"] = self.key
            # Authenticate
            try:
                kaggle.api.authenticate()
            except Exception as e:
                print(f"Warning: Kaggle authentication failed: {e}")
    
    def download_dataset(
        self,
        dataset_name: str,
        output_dir: Optional[Path] = None,
        unzip: bool = True
    ) -> Path:
        """
        Download a Kaggle dataset.
        
        Args:
            dataset_name: Dataset name in format 'username/dataset-name'
            output_dir: Directory to save dataset (default: settings.data_dir)
            unzip: Whether to unzip downloaded files
            
        Returns:
            Path to downloaded dataset directory
        """
        if output_dir is None:
            output_dir = settings.data_dir / "kaggle" / dataset_name.replace("/", "_")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(output_dir),
                unzip=unzip
            )
            
            return output_dir
            
        except Exception as e:
            raise ValueError(f"Failed to download Kaggle dataset {dataset_name}: {e}")
    
    def download_competition(
        self,
        competition_name: str,
        output_dir: Optional[Path] = None,
        unzip: bool = True
    ) -> Path:
        """
        Download a Kaggle competition dataset.
        
        Args:
            competition_name: Competition name
            output_dir: Directory to save dataset (default: settings.data_dir)
            unzip: Whether to unzip downloaded files
            
        Returns:
            Path to downloaded dataset directory
        """
        if output_dir is None:
            output_dir = settings.data_dir / "kaggle" / "competitions" / competition_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download competition files
            kaggle.api.competition_download_files(
                competition_name,
                path=str(output_dir),
                unzip=unzip
            )
            
            return output_dir
            
        except Exception as e:
            raise ValueError(f"Failed to download Kaggle competition {competition_name}: {e}")
    
    def load_dataset_files(
        self,
        dataset_path: Path,
        file_extensions: Optional[list] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load dataset files from a directory.
        
        Args:
            dataset_path: Path to dataset directory
            file_extensions: List of file extensions to load (default: ['.csv', '.xlsx', '.xls'])
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        if file_extensions is None:
            file_extensions = ['.csv', '.xlsx', '.xls']
        
        datasets = {}
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Find all matching files
        for ext in file_extensions:
            for file_path in dataset_path.glob(f"*{ext}"):
                try:
                    if ext == '.csv':
                        df = pd.read_csv(file_path)
                    elif ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path)
                    else:
                        continue
                    
                    # Use filename without extension as key
                    key = file_path.stem
                    datasets[key] = df
                    
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue
        
        return datasets
    
    def get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get metadata for a dataset.
        
        Args:
            dataset_name: Dataset name in format 'username/dataset-name'
            
        Returns:
            Dictionary with dataset metadata
        """
        try:
            dataset = kaggle.api.dataset_view(dataset_name)
            return {
                "source": "kaggle",
                "name": dataset_name,
                "title": dataset.get("title", ""),
                "description": dataset.get("description", ""),
                "size": dataset.get("size", 0),
                "fileCount": dataset.get("fileCount", 0),
                "tags": dataset.get("tags", []),
            }
        except Exception as e:
            print(f"Warning: Failed to get metadata for {dataset_name}: {e}")
            return {
                "source": "kaggle",
                "name": dataset_name,
            }

