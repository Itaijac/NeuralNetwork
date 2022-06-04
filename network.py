import numpy as np
import os

MINI_BATCH_SIZE = 32


class Network:
    """
    This class represents the Neural Network and takes care of all
    associated functions, such as train or predict.
    """

    def __init__(self, *args):
        """
        Constructor
        """
        if len(args) == 0:
            self.layers = []
        else:
            self.layers = args[0]

    def add(self, layer):
        """
        Adds a layer to the Neural Network.
        """
        self.layers.append(layer)

    def train(self, x_train, y_train, epoch, verbose=True):
        """
        This function trains the Neural Network by showcasing a bunch
        of digits to it over and over until the Neural Network
        will be able to tell which digit is which by itself.

        <Stochastic Gradient Descent>
        Stochastic Gradient Descent is a method of gradient descent
        that is used to train a model in which the training data is
        presented in random order.

        <Mini-Batch Gradient Descent>
        Mini-batch gradient descent is a method of gradient descent
        that is used to train a model in which the training data is
        presented in mini-batches.

        In this train function we use both gradient descent methods.
        """
        trainable_parameters = 0
        for layer in self.layers:
            # Calculate the number of trainable parameters for the layer
            trainable_parameters += len(layer.weights.flatten()) + \
                len(layer.bias.flatten())

        for i in range(epoch):
            mse = 0

            # Training the model with the data in mini-batches
            for j in range(round(len(x_train)/MINI_BATCH_SIZE)):
                mini_batch_x = x_train[j*MINI_BATCH_SIZE:j *
                                       MINI_BATCH_SIZE+MINI_BATCH_SIZE]
                mini_batch_y = y_train[j*MINI_BATCH_SIZE:j *
                                       MINI_BATCH_SIZE+MINI_BATCH_SIZE]
                overall_gradient_vector = np.empty(
                    [MINI_BATCH_SIZE, trainable_parameters])

                # Iterate over the mini-batch
                for index, (x, y) in enumerate(zip(mini_batch_x, mini_batch_y)):
                    gradient_vector = np.array([])

                    # Forward Propagation
                    for layer in self.layers:
                        x = layer.forward_propagation(x)

                    # Calculate Cost
                    output_data = self.calculate_cost(y, x)

                    # Mean Squared Error
                    mse += np.mean(np.square(x - y))

                    # Backward Propagation
                    for layer in reversed(self.layers):
                        # Calculate the gradient vector
                        output_data, weight_gradient, bias_gradient = layer.backward_propagation(
                            output_data)
                        gradient_vector = np.concatenate(
                            (gradient_vector, weight_gradient.flatten(), bias_gradient.flatten()))
                    # Append the gradient vector to the overall gradient vector
                    overall_gradient_vector[index] = gradient_vector

                self.update_parameters(overall_gradient_vector)

            mse /= len(x_train)
            if verbose:
                print(
                    f'Epoch {i+1}/{epoch} [{"=" * (i)}>{"." * (epoch-i-1)}] | Mean Squared Error: {mse}')

    def update_parameters(self, overall_gradient_vector):
        overall_gradient_vector = np.mean(overall_gradient_vector, axis=0)
        for layer in reversed(self.layers):
            weights_len = len(layer.weights.flatten())
            bias_len = len(layer.bias.flatten())
            layer.weights -= overall_gradient_vector[:weights_len].reshape(
                layer.weights.shape)
            layer.bias -= overall_gradient_vector[weights_len:weights_len + bias_len].reshape(
                layer.bias.shape)
            overall_gradient_vector = overall_gradient_vector[weights_len + bias_len:]

    def calculate_cost(self, y_true, y_prediction):
        """
        This function calculates the cost of the Neural Network.
        """
        return (2 * (y_prediction - y_true)).flatten()

    def predict(self, input_data):
        """
        This function predicts the digit of the input_data.
        """
        output_data = input_data
        for layer in self.layers:
            output_data = layer.forward_propagation(output_data)

        # Calculate Probablity of each digit
        output_data = output_data.flatten()
        output_data_sum = sum(output_data)
        probablity = [probablity /
                      output_data_sum for probablity in output_data]

        # Return the digit with the highest probability
        return np.argmax(probablity), np.amax(probablity), probablity

    def save(self, path):
        """
        Saves the model to the PATH directory.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        for index, layer in enumerate(self.layers):
            np.save(f'{path}\\{index}_weights.npy', layer.weights)
            np.save(f'{path}\\{index}_bias.npy', layer.bias)

    def load(self, path):
        """
        Loads the model from the PATH directory.
        """
        for index, layer in enumerate(self.layers):
            layer.weights = np.load(f'{path}\\{index}_weights.npy')
            layer.bias = np.load(f'{path}\\{index}_bias.npy')

    def display(self):
        """
        This functions prints the layers of the network.
        """
        for index, layer in enumerate(self.layers):
            print(f'Layer {index}:\n{layer}\n')
