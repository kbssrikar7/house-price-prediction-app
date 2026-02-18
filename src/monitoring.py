"""
Model Monitoring and Drift Detection.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger("ml_portfolio")

def monitor_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path,
    column_mapping: Optional[dict] = None
) -> None:
    """
    Generate a data drift report comparing reference and current data.
    
    Args:
        reference_data: Training data (baseline)
        current_data: Production/Test data (new batch)
        output_path: Path to save HTML report
        column_mapping: Optional Evidently column mapping
    """
    logger.info("Starting drift detection...")
    
    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    logger.info(f"Reference shape: {reference_data.shape}, Current shape: {current_data.shape}")
    
    # Run drift calculation (Evidently API: run(current_data, reference_data) -> Snapshot)
    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )
    
    # Save report (Snapshot has save_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    
    logger.info(f"Drift report saved to {output_path}")

    # Check for drift (basic check from JSON export if needed, 
    # but HTML is sufficient for this portfolio)
    # result = report.as_dict()
    # drift_share = result['metrics'][0]['result']['drift_share']
    # if drift_share > 0.5:
    #     logger.warning(f"High data drift detected! Share: {drift_share}")

if __name__ == "__main__":
    # Test run
    # Mock data
    ref = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    curr = pd.DataFrame({"a": [1, 2, 10], "b": [4, 5, 6]})
    monitor_drift(ref, curr, Path("reports/drift_test.html"))
