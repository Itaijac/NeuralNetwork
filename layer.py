import numpy as np


class Layer:
    """
    Creates a fully connected activation layer.
    """

    def __init__(self, input_size, output_size, activation, activation_prime):
        """
        Constructor
        """
        self.input = None
        self.output = None

        # Add the activation function and its derivative
        self.activation = activation
        self.activation_prime = activation_prime

        # Initialize weights and bias, and change the distribution from
        # 0 to 1, to -1 to 1.
        self.weights = (np.random.rand(input_size, output_size) - 0.5) * 2
        self.bias = (np.random.rand(1, output_size) - 0.5) * 2

    def forward_propagation(self, input_data):
        """
        This function calculates the activation of a the Neural layer
        given the input_data
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.activation(self.output)

    def backward_propagation(self, error):
        """
        This function calculates the error of the Neural layer given the
        error of the next layer. With the error of the next layer, the
        weights and the bias of the current layer are updated.
        """
        # Calculate input, weight and bias gradient
        error = error * self.activation_prime(self.output)
        input_gradient = np.dot(error, self.weights.T)
        weight_gradient = np.dot(self.input.T, error)
        bias_gradient = error

        return input_gradient, weight_gradient, bias_gradient

    def __str__(self):
        return "Fully Connected Layer"
