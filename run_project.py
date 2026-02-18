"""
Project Execution Orchestrator.
Launches MLflow, FastAPI, and Streamlit services simultaneously.
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

def stream_logs(process, name):
    """Stream logs from a process to stdout."""
    # In a real scenario, we might want to thread this or logging logic
    # For now, we rely on the process outputting to console if not captured
    pass

def main():
    print("Starting ML Portfolio Project Services...")
    
    processes = []
    
    try:
        # 1. Start MLflow UI
        print("Starting MLflow UI on port 5000...")
        mlflow_process = subprocess.Popen(
            ["uv", "run", "mlflow", "ui", "--port", "5000"],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("MLflow", mlflow_process))
        
        # 2. Start FastAPI Backend
        print("Starting FastAPI Backend on port 8000...")
        fastapi_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("FastAPI", fastapi_process))
        
        # 3. Start Streamlit Dashboard
        print("Starting Streamlit Dashboard on port 8501...")
        streamlit_process = subprocess.Popen(
            ["uv", "run", "streamlit", "run", "src/dashboard.py", "--server.port", "8501", "--server.headless", "true"],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("Streamlit", streamlit_process))
        
        print("\n[OK] All services running!")
        print("   - MLflow UI: http://localhost:5000")
        print("   - FastAPI API: http://localhost:8000")
        print("   - Swagger Docs: http://localhost:8000/docs")
        print("   - Dashboard: http://localhost:8501")
        print("\nPress Ctrl+C to stop all services.")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, p in processes:
                if p.poll() is not None:
                    print(f"\n[WARN] {name} has stopped (Exit Code: {p.returncode})")
                    raise KeyboardInterrupt # Trigger cleanup
                    
    except KeyboardInterrupt:
        print("\n\nStopping services...")
    finally:
        for name, p in processes:
            if p.poll() is None:
                print(f"Terminating {name}...")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
