"""
Core modules for Fire Dynamics Physics-Informed Neural Networks
"""

from .neural_network import NeuralNetwork, FirePINN, Activation
from .training import PINNTrainer, generate_training_data, generate_boundary_data

__all__ = [
    'NeuralNetwork',
    'FirePINN',
    'Activation',
    'PINNTrainer',
    'generate_training_data',
    'generate_boundary_data'
]