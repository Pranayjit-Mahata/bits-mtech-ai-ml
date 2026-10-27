"""
Module: gradient_descent_1d.py
Course Reference: Mathematical Foundations for ML (MFML)
Description: 
    A clean, from-scratch implementation of the Gradient Descent optimization 
    algorithm applied to a 1D quadratic cost function: f(x) = x^2 - 4x + 4.
    The global minimum of this function is at x = 2, where f(x) = 0.
"""

import numpy as np


def cost_function(x: float) -> float:
    """
    Computes the value of the objective function f(x) = x^2 - 4x + 4.
    """
    return x**2 - 4 * x + 4


def compute_gradient(x: float) -> float:
    """
    Computes the first derivative (gradient) of the cost function.
    f'(x) = 2x - 4
    """
    return 2 * x - 4


def run_gradient_descent(
    initial_x: float, learning_rate: float, epochs: int, tolerance: float = 1e-6
) -> float:
    """
    Executes the gradient descent optimization loop.

    Parameters:
        initial_x (float): The starting point of the optimization parameter.
        learning_rate (float): Step size factor (alpha).
        epochs (int): Maximum number of iterations to perform.
        tolerance (float): Threshold to stop early if the gradient becomes negligible.

    Returns:
        float: The optimized value of x that minimizes the function.
    """
    current_x = initial_x

    print(f"Starting Gradient Descent at x = {current_x:.4f}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # 1. Compute the gradient at the current position
        gradient = compute_gradient(current_x)

        # Early stopping condition if the gradient is close to zero
        if abs(gradient) < tolerance:
            print(f"Converged early at Epoch {epoch}: Gradient is near zero.")
            break

        # 2. Update rule: Move in the opposite direction of the gradient
        # x_new = x_old - (learning_rate * gradient)
        current_x = current_x - (learning_rate * gradient)

        # 3. Log progress every 5 epochs to keep outputs clean
        if epoch % 5 == 0 or epoch == 1:
            current_cost = cost_function(current_x)
            print(
                f"Epoch {epoch:02d} | x = {current_x:.6f} | Cost = {current_cost:.6f} | Gradient = {gradient:.6f}"
            )

    print("-" * 50)
    print(f"Optimization Complete. Final x: {current_x:.6f}")
    return current_x


if __name__ == "__main__":
    # Hyperparameters
    STARTING_POINT = 10.0  # Start far away from the minimum (x=2)
    LEARNING_RATE = 0.1  # Step size factor
    MAX_EPOCHS = 40  # Iteration limit

    # Run the optimizer
    optimized_x = run_gradient_descent(
        initial_x=STARTING_POINT, learning_rate=LEARNING_RATE, epochs=MAX_EPOCHS
    )
