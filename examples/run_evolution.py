#!/usr/bin/env python3

"""
Main script to run the evolutionary robotics simulation for the hexapod robot.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulation import main

if __name__ == "__main__":
    main()