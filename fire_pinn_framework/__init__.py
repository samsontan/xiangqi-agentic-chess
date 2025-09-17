"""
Fire Dynamics Physics-Informed Neural Networks (PINNs) Framework

A lightweight implementation designed to replace traditional CFD methods
like CFAST/FDS with neural network-based solutions for fire risk assessment.

This framework provides:
- Physics-informed neural networks for fire dynamics
- Heat transfer equation solvers
- Navier-Stokes approximations for smoke transport
- Integration interfaces for AAMKS fire risk assessment

Author: Claude AI Assistant
Date: 2025-09-17
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"

# Framework modules
from .core import *
from .models import *
from .utils import *