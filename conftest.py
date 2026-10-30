"""
Pytest root configuration: ensures the workspace root is always at the head of sys.path.
"""
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
