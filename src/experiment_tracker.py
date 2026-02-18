"""MLflow experiment tracking integration."""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class ExperimentTracker:
    """Wrapper for MLflow experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str = "ml-portfolio-project",
        tracking_uri: str = "mlruns"
    ):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server or local directory
            # Note: For local runs, passing 'mlruns' as URI works if it's a path or file: URI
            # But mlflow default is already ./mlruns if not set.
        """
        self.experiment_name = experiment_name
        # normalize tracking URI if local path
        if not tracking_uri.startswith("http") and not tracking_uri.startswith("file:"):
             tracking_uri = f"file:///{Path(tracking_uri).absolute()}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        return self.run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        # Flatten nested dicts - basic support
        # MLflow handles some nesting but flattening is safer
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log metrics to current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log a model to current run.
        
        Args:
            model: Trained model object
            artifact_path: Path within artifacts to save model
            registered_model_name: Optional name to register model
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Optional path within artifacts directory
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the artifact
        """
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
