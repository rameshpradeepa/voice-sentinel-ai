"""
VoiceSentinel AI - Pytest Configuration File
---------------------------------------------
Ensures project root directory is added to sys.path during pytest module collection.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
