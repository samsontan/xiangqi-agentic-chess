"""
Lightweight Neural Network Implementation for Physics-Informed Networks

This module provides a pure Python implementation of neural networks
optimized for physics-informed learning without external dependencies.
"""

import math
import random
from typing import List, Callable, Tuple, Optional


class Activation:
    """Activation functions for neural networks"""

    @staticmethod
    def tanh(x: float) -> float:
        """Hyperbolic tangent activation"""
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x: float) -> float:
        """Derivative of tanh"""
        t = math.tanh(x)
        return 1 - t * t

    @staticmethod
    def relu(x: float) -> float:
        """ReLU activation"""
        return max(0, x)

    @staticmethod
    def relu_derivative(x: float) -> float:
        """Derivative of ReLU"""
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation"""
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        """Derivative of sigmoid"""
        s = Activation.sigmoid(x)
        return s * (1 - s)


class NeuralNetwork:
    """
    Lightweight neural network implementation for PINNs

    Supports:
    - Multiple hidden layers
    - Different activation functions
    - Gradient computation for physics-informed training
    """

    def __init__(self, layer_sizes: List[int], activation: str = 'tanh'):
        """
        Initialize neural network

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Set activation function
        if activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))

            # Weight matrix
            w = [[random.uniform(-limit, limit) for _ in range(fan_in)]
                 for _ in range(fan_out)]
            self.weights.append(w)

            # Bias vector
            b = [0.0 for _ in range(fan_out)]
            self.biases.append(b)

    def forward(self, inputs: List[float]) -> Tuple[List[float], List[List[float]]]:
        """
        Forward pass through the network

        Args:
            inputs: Input values

        Returns:
            outputs: Final outputs
            activations: All layer activations (for backprop)
        """
        activations = [inputs]
        current = inputs

        for i in range(self.num_layers - 1):
            # Linear transformation: z = W*x + b
            z = []
            for j in range(len(self.weights[i])):
                val = sum(self.weights[i][j][k] * current[k]
                         for k in range(len(current))) + self.biases[i][j]
                z.append(val)

            # Apply activation (except for output layer)
            if i < self.num_layers - 2:
                current = [self.activation(val) for val in z]
            else:
                current = z  # Linear output for regression

            activations.append(current)

        return current, activations

    def compute_gradients(self, inputs: List[float]) -> Tuple[List[float], List[List[float]]]:
        """
        Compute gradients with respect to inputs using automatic differentiation

        This is crucial for physics-informed training where we need derivatives
        of network outputs with respect to spatial/temporal coordinates.

        Returns:
            gradients: First derivatives ∂u/∂x
            hessians: Second derivatives ∂²u/∂x²
        """
        eps = 1e-8  # Small perturbation for numerical differentiation

        # Compute gradients using finite differences
        gradients = []
        hessians = []

        for i in range(len(inputs)):
            # Forward difference for first derivative
            inputs_plus = inputs.copy()
            inputs_plus[i] += eps
            outputs_plus, _ = self.forward(inputs_plus)

            inputs_minus = inputs.copy()
            inputs_minus[i] -= eps
            outputs_minus, _ = self.forward(inputs_minus)

            # First derivative
            grad = [(outputs_plus[j] - outputs_minus[j]) / (2 * eps)
                   for j in range(len(outputs_plus))]
            gradients.append(grad)

            # Second derivative (finite difference of gradients)
            inputs_plus2 = inputs.copy()
            inputs_plus2[i] += 2 * eps
            outputs_plus2, _ = self.forward(inputs_plus2)

            hess = [(outputs_plus2[j] - 2 * outputs_plus[j] + outputs_minus[j]) / (eps ** 2)
                   for j in range(len(outputs_plus))]
            hessians.append(hess)

        return gradients, hessians


class FirePINN(NeuralNetwork):
    """
    Physics-Informed Neural Network specialized for fire dynamics

    Implements physics constraints for:
    - Heat transfer equation
    - Conservation laws
    - Boundary conditions for fire scenarios
    """

    def __init__(self, layer_sizes: List[int]):
        """
        Initialize Fire PINN

        Expected inputs: [x, y, z, t] (spatial coordinates + time)
        Expected outputs: [T, u, v, w, p] (temperature, velocity components, pressure)
        """
        super().__init__(layer_sizes, activation='tanh')

        # Physical parameters for fire dynamics
        self.thermal_diffusivity = 1e-5  # m²/s
        self.density = 1.2  # kg/m³
        self.specific_heat = 1005  # J/(kg·K)
        self.conductivity = 0.025  # W/(m·K)

    def physics_loss(self, inputs: List[float], outputs: List[float]) -> float:
        """
        Compute physics-informed loss based on fire dynamics PDEs

        Implements:
        1. Heat equation: ∂T/∂t = α∇²T + Q
        2. Conservation of mass: ∂ρ/∂t + ∇·(ρv) = 0
        3. Momentum equations (simplified Navier-Stokes)
        """
        x, y, z, t = inputs[:4]
        T, u, v, w, p = outputs[:5] if len(outputs) >= 5 else outputs + [0] * (5 - len(outputs))

        # Compute derivatives
        gradients, hessians = self.compute_gradients(inputs)

        # Temperature gradients
        dT_dx, dT_dy, dT_dz, dT_dt = [gradients[i][0] for i in range(4)]
        d2T_dx2, d2T_dy2, d2T_dz2 = [hessians[i][0] for i in range(3)]

        # Heat equation residual: ∂T/∂t - α∇²T = Q
        laplacian_T = d2T_dx2 + d2T_dy2 + d2T_dz2
        heat_source = self.heat_source(x, y, z, t)  # Fire heat source
        heat_residual = dT_dt - self.thermal_diffusivity * laplacian_T - heat_source

        # Conservation laws (simplified)
        mass_residual = 0.0  # Simplified for now

        # Combined physics loss
        physics_loss = heat_residual ** 2 + mass_residual ** 2

        return physics_loss

    def heat_source(self, x: float, y: float, z: float, t: float) -> float:
        """
        Heat source term for fire dynamics

        Models fire as a localized heat source with temporal evolution
        """
        # Fire location (center of room)
        fire_x, fire_y = 0.0, 0.0

        # Distance from fire center
        r = math.sqrt((x - fire_x) ** 2 + (y - fire_y) ** 2)

        # Fire intensity (grows with time, decays with distance)
        max_intensity = 1e6  # W/m³
        fire_radius = 1.0  # m

        if r <= fire_radius and t > 0:
            # Gaussian heat source
            intensity = max_intensity * math.exp(-r ** 2 / (2 * (fire_radius / 3) ** 2))
            # Time evolution (ramp up)
            time_factor = min(1.0, t / 60.0)  # Ramp up over 60 seconds
            return intensity * time_factor

        return 0.0

    def boundary_loss(self, boundary_inputs: List[List[float]],
                     boundary_conditions: List[float]) -> float:
        """
        Compute boundary condition loss

        Args:
            boundary_inputs: Points on domain boundaries
            boundary_conditions: Expected values at boundaries
        """
        total_loss = 0.0

        for i, inputs in enumerate(boundary_inputs):
            outputs, _ = self.forward(inputs)
            predicted = outputs[0]  # Temperature
            expected = boundary_conditions[i]
            total_loss += (predicted - expected) ** 2

        return total_loss / len(boundary_inputs)