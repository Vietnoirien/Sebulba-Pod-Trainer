#!/usr/bin/env python3
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sebulba_pod_trainer.environment.heuristic_test_visualizer import main

if __name__ == "__main__":
    main()