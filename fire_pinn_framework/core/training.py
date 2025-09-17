"""
Training module for Physics-Informed Neural Networks

Implements gradient-based optimization specifically for fire dynamics PINNs
"""

import math
import random
from typing import List, Tuple, Callable
from .neural_network import FirePINN


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks

    Implements Adam optimizer and physics-informed loss computation
    """

    def __init__(self, model: FirePINN, learning_rate: float = 0.001):
        """
        Initialize PINN trainer

        Args:
            model: FirePINN model to train
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.learning_rate = learning_rate

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # Initialize momentum and velocity for Adam
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        for i in range(len(model.weights)):
            # Weights
            m_w = [[0.0 for _ in range(len(model.weights[i][j]))]
                   for j in range(len(model.weights[i]))]
            v_w = [[0.0 for _ in range(len(model.weights[i][j]))]
                   for j in range(len(model.weights[i]))]
            self.m_weights.append(m_w)
            self.v_weights.append(v_w)

            # Biases
            m_b = [0.0 for _ in range(len(model.biases[i]))]
            v_b = [0.0 for _ in range(len(model.biases[i]))]
            self.m_biases.append(m_b)
            self.v_biases.append(v_b)

        self.t = 0  # Time step for Adam

    def compute_gradients(self, data_points: List[List[float]],
                         boundary_points: List[List[float]],
                         boundary_values: List[float],
                         weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[List, List]:
        """
        Compute gradients for physics-informed training

        Args:
            data_points: Interior domain points for physics loss
            boundary_points: Boundary points for boundary loss
            boundary_values: Expected values at boundaries
            weights: (data_weight, physics_weight, boundary_weight)

        Returns:
            weight_gradients: Gradients w.r.t. weights
            bias_gradients: Gradients w.r.t. biases
        """
        data_weight, physics_weight, boundary_weight = weights

        # Initialize gradient accumulators
        weight_gradients = []
        bias_gradients = []

        for i in range(len(self.model.weights)):
            w_grad = [[0.0 for _ in range(len(self.model.weights[i][j]))]
                     for j in range(len(self.model.weights[i]))]
            b_grad = [0.0 for _ in range(len(self.model.biases[i]))]
            weight_gradients.append(w_grad)
            bias_gradients.append(b_grad)

        total_loss = 0.0
        num_points = len(data_points)

        # Compute gradients using finite differences
        eps = 1e-6

        # For each weight parameter
        for layer_idx in range(len(self.model.weights)):
            for i in range(len(self.model.weights[layer_idx])):
                for j in range(len(self.model.weights[layer_idx][i])):
                    # Perturb weight
                    self.model.weights[layer_idx][i][j] += eps
                    loss_plus = self.compute_total_loss(data_points, boundary_points,
                                                       boundary_values, weights)

                    self.model.weights[layer_idx][i][j] -= 2 * eps
                    loss_minus = self.compute_total_loss(data_points, boundary_points,
                                                        boundary_values, weights)

                    # Restore weight
                    self.model.weights[layer_idx][i][j] += eps

                    # Compute gradient
                    gradient = (loss_plus - loss_minus) / (2 * eps)
                    weight_gradients[layer_idx][i][j] = gradient

        # For each bias parameter
        for layer_idx in range(len(self.model.biases)):
            for i in range(len(self.model.biases[layer_idx])):
                # Perturb bias
                self.model.biases[layer_idx][i] += eps
                loss_plus = self.compute_total_loss(data_points, boundary_points,
                                                   boundary_values, weights)

                self.model.biases[layer_idx][i] -= 2 * eps
                loss_minus = self.compute_total_loss(data_points, boundary_points,
                                                    boundary_values, weights)

                # Restore bias
                self.model.biases[layer_idx][i] += eps

                # Compute gradient
                gradient = (loss_plus - loss_minus) / (2 * eps)
                bias_gradients[layer_idx][i] = gradient

        return weight_gradients, bias_gradients

    def compute_total_loss(self, data_points: List[List[float]],
                          boundary_points: List[List[float]],
                          boundary_values: List[float],
                          weights: Tuple[float, float, float]) -> float:
        """
        Compute total loss combining data, physics, and boundary terms
        """
        data_weight, physics_weight, boundary_weight = weights

        total_loss = 0.0

        # Physics loss on interior points
        physics_loss = 0.0
        for point in data_points:
            outputs, _ = self.model.forward(point)
            physics_loss += self.model.physics_loss(point, outputs)
        physics_loss /= len(data_points)

        # Boundary loss
        boundary_loss = self.model.boundary_loss(boundary_points, boundary_values)

        # Combined loss
        total_loss = physics_weight * physics_loss + boundary_weight * boundary_loss

        return total_loss

    def adam_update(self, weight_gradients: List, bias_gradients: List):
        """
        Adam optimizer update step
        """
        self.t += 1

        # Bias correction terms
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        # Update weights
        for layer_idx in range(len(self.model.weights)):
            for i in range(len(self.model.weights[layer_idx])):
                for j in range(len(self.model.weights[layer_idx][i])):
                    g = weight_gradients[layer_idx][i][j]

                    # Update moments
                    self.m_weights[layer_idx][i][j] = (self.beta1 * self.m_weights[layer_idx][i][j] +
                                                      (1 - self.beta1) * g)
                    self.v_weights[layer_idx][i][j] = (self.beta2 * self.v_weights[layer_idx][i][j] +
                                                      (1 - self.beta2) * g * g)

                    # Update weight
                    self.model.weights[layer_idx][i][j] -= (lr_t * self.m_weights[layer_idx][i][j] /
                                                           (math.sqrt(self.v_weights[layer_idx][i][j]) + self.epsilon))

        # Update biases
        for layer_idx in range(len(self.model.biases)):
            for i in range(len(self.model.biases[layer_idx])):
                g = bias_gradients[layer_idx][i]

                # Update moments
                self.m_biases[layer_idx][i] = (self.beta1 * self.m_biases[layer_idx][i] +
                                              (1 - self.beta1) * g)
                self.v_biases[layer_idx][i] = (self.beta2 * self.v_biases[layer_idx][i] +
                                              (1 - self.beta2) * g * g)

                # Update bias
                self.model.biases[layer_idx][i] -= (lr_t * self.m_biases[layer_idx][i] /
                                                   (math.sqrt(self.v_biases[layer_idx][i]) + self.epsilon))

    def train_step(self, data_points: List[List[float]],
                   boundary_points: List[List[float]],
                   boundary_values: List[float],
                   weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Single training step

        Returns:
            Current loss value
        """
        # Compute gradients
        weight_grads, bias_grads = self.compute_gradients(data_points, boundary_points,
                                                         boundary_values, weights)

        # Update parameters
        self.adam_update(weight_grads, bias_grads)

        # Return current loss
        return self.compute_total_loss(data_points, boundary_points, boundary_values, weights)

    def train(self, data_points: List[List[float]],
              boundary_points: List[List[float]],
              boundary_values: List[float],
              epochs: int = 1000,
              weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
              verbose: bool = True) -> List[float]:
        """
        Train the PINN model

        Args:
            data_points: Interior domain points
            boundary_points: Boundary points
            boundary_values: Expected boundary values
            epochs: Number of training epochs
            weights: Loss term weights
            verbose: Print training progress

        Returns:
            List of loss values during training
        """
        loss_history = []

        for epoch in range(epochs):
            loss = self.train_step(data_points, boundary_points, boundary_values, weights)
            loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return loss_history


def generate_training_data(domain_bounds: Tuple[Tuple[float, float], ...],
                          n_points: int = 1000) -> List[List[float]]:
    """
    Generate random training points in the domain

    Args:
        domain_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max), (t_min, t_max))
        n_points: Number of points to generate

    Returns:
        List of [x, y, z, t] points
    """
    points = []

    for _ in range(n_points):
        point = []
        for (min_val, max_val) in domain_bounds:
            point.append(random.uniform(min_val, max_val))
        points.append(point)

    return points


def generate_boundary_data(domain_bounds: Tuple[Tuple[float, float], ...],
                          n_points_per_face: int = 100) -> Tuple[List[List[float]], List[float]]:
    """
    Generate boundary points and corresponding values

    Args:
        domain_bounds: Domain boundaries
        n_points_per_face: Points per boundary face

    Returns:
        boundary_points: Points on boundaries
        boundary_values: Expected values (temperature = ambient for walls)
    """
    boundary_points = []
    boundary_values = []

    (x_min, x_max), (y_min, y_max), (z_min, z_max), (t_min, t_max) = domain_bounds

    # Ambient temperature
    T_ambient = 293.15  # 20°C in Kelvin

    # Generate points on each boundary face
    faces = [
        (x_min, 'x'),  # Left wall
        (x_max, 'x'),  # Right wall
        (y_min, 'y'),  # Front wall
        (y_max, 'y'),  # Back wall
        (z_min, 'z'),  # Floor
        (z_max, 'z'),  # Ceiling
    ]

    for face_value, face_dim in faces:
        for _ in range(n_points_per_face):
            if face_dim == 'x':
                point = [face_value,
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max),
                        random.uniform(t_min, t_max)]
            elif face_dim == 'y':
                point = [random.uniform(x_min, x_max),
                        face_value,
                        random.uniform(z_min, z_max),
                        random.uniform(t_min, t_max)]
            else:  # face_dim == 'z'
                point = [random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        face_value,
                        random.uniform(t_min, t_max)]

            boundary_points.append(point)
            boundary_values.append(T_ambient)

    return boundary_points, boundary_values