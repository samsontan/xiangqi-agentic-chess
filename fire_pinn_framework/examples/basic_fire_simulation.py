#!/usr/bin/env python3
"""
Basic Fire Simulation using Physics-Informed Neural Networks

This example demonstrates how to use the Fire PINN framework to solve
a simplified fire dynamics problem in a room.

Usage:
    python basic_fire_simulation.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import FirePINN, PINNTrainer, generate_training_data, generate_boundary_data


def main():
    """
    Run a basic fire simulation using PINNs
    """
    print("=== Fire Dynamics PINN Simulation ===")
    print("Setting up physics-informed neural network for fire dynamics...")

    # Define problem domain (room dimensions + time)
    # Room: 5m x 5m x 3m, simulation time: 0-300 seconds
    domain_bounds = (
        (-2.5, 2.5),  # x: -2.5m to 2.5m
        (-2.5, 2.5),  # y: -2.5m to 2.5m
        (0.0, 3.0),   # z: 0m to 3m (floor to ceiling)
        (0.0, 300.0)  # t: 0 to 300 seconds
    )

    # Create Fire PINN model
    # Input: [x, y, z, t] (4D)
    # Output: [T] (temperature field)
    layer_sizes = [4, 32, 32, 32, 1]  # 4 hidden layers
    model = FirePINN(layer_sizes)

    print(f"Created Fire PINN with architecture: {layer_sizes}")
    print(f"Total parameters: {sum(len(layer) * len(layer[0]) for layer in model.weights) + sum(len(bias) for bias in model.biases)}")

    # Generate training data
    print("Generating training data...")
    n_interior = 1000
    n_boundary_per_face = 50

    # Interior points for physics loss
    interior_points = generate_training_data(domain_bounds, n_interior)

    # Boundary points and conditions
    boundary_points, boundary_values = generate_boundary_data(domain_bounds, n_boundary_per_face)

    print(f"Generated {len(interior_points)} interior points")
    print(f"Generated {len(boundary_points)} boundary points")

    # Create trainer
    trainer = PINNTrainer(model, learning_rate=0.001)

    # Train the model
    print("Starting training...")
    epochs = 500
    loss_weights = (0.0, 1.0, 1.0)  # (data_weight, physics_weight, boundary_weight)

    loss_history = trainer.train(
        data_points=interior_points,
        boundary_points=boundary_points,
        boundary_values=boundary_values,
        epochs=epochs,
        weights=loss_weights,
        verbose=True
    )

    print(f"Training completed. Final loss: {loss_history[-1]:.6f}")

    # Test the trained model
    print("\n=== Testing Trained Model ===")
    test_predictions(model, domain_bounds)

    # Save model (simplified save)
    print("\n=== Saving Model ===")
    save_model(model, "fire_pinn_model.txt")

    print("Simulation completed successfully!")


def test_predictions(model: FirePINN, domain_bounds):
    """
    Test the trained model with specific scenarios
    """
    print("Testing model predictions...")

    # Test points at different locations and times
    test_cases = [
        # [x, y, z, t] - description
        ([0.0, 0.0, 1.5, 60.0], "Fire center, 1.5m height, after 1 min"),
        ([1.0, 1.0, 1.5, 60.0], "1m from fire, 1.5m height, after 1 min"),
        ([2.0, 2.0, 1.5, 60.0], "Corner, 1.5m height, after 1 min"),
        ([0.0, 0.0, 2.5, 120.0], "Fire center, near ceiling, after 2 min"),
        ([0.0, 0.0, 0.5, 180.0], "Fire center, near floor, after 3 min"),
    ]

    for point, description in test_cases:
        outputs, _ = model.forward(point)
        temperature = outputs[0]

        # Convert to Celsius for display
        temp_celsius = temperature - 273.15

        print(f"{description}: {temp_celsius:.1f}°C")

    # Test temperature evolution over time at fire center
    print("\nTemperature evolution at fire center (0,0,1.5m):")
    times = [0, 30, 60, 120, 180, 240, 300]
    for t in times:
        point = [0.0, 0.0, 1.5, float(t)]
        outputs, _ = model.forward(point)
        temp_celsius = outputs[0] - 273.15
        print(f"t={t:3d}s: {temp_celsius:.1f}°C")


def save_model(model: FirePINN, filename: str):
    """
    Save model weights and biases to a text file
    """
    try:
        with open(filename, 'w') as f:
            f.write("# Fire PINN Model\n")
            f.write(f"# Architecture: {model.layer_sizes}\n")
            f.write(f"# Thermal diffusivity: {model.thermal_diffusivity}\n\n")

            # Save weights
            f.write("WEIGHTS:\n")
            for i, layer_weights in enumerate(model.weights):
                f.write(f"Layer {i}:\n")
                for j, neuron_weights in enumerate(layer_weights):
                    f.write(f"  Neuron {j}: {neuron_weights}\n")

            # Save biases
            f.write("\nBIASES:\n")
            for i, layer_biases in enumerate(model.biases):
                f.write(f"Layer {i}: {layer_biases}\n")

        print(f"Model saved to {filename}")

    except Exception as e:
        print(f"Error saving model: {e}")


def demonstrate_physics_compliance():
    """
    Demonstrate that the PINN respects physical laws
    """
    print("\n=== Physics Compliance Demonstration ===")

    # Create a simple model for demonstration
    model = FirePINN([4, 16, 16, 1])

    # Test point near fire
    point = [0.0, 0.0, 1.0, 60.0]  # Fire center, 1m height, 60s
    outputs, _ = model.forward(point)

    # Compute physics residuals
    physics_loss = model.physics_loss(point, outputs)
    print(f"Physics residual at fire center: {physics_loss:.6f}")

    # Test heat source function
    heat_at_fire = model.heat_source(0.0, 0.0, 1.0, 60.0)
    heat_away_from_fire = model.heat_source(3.0, 3.0, 1.0, 60.0)

    print(f"Heat source at fire center: {heat_at_fire:.2e} W/m³")
    print(f"Heat source 3m away: {heat_away_from_fire:.2e} W/m³")

    # Demonstrate boundary conditions
    boundary_point = [2.5, 0.0, 1.0, 60.0]  # Wall boundary
    wall_outputs, _ = model.forward(boundary_point)
    print(f"Temperature at wall boundary: {wall_outputs[0]:.1f} K")


if __name__ == "__main__":
    main()
    demonstrate_physics_compliance()