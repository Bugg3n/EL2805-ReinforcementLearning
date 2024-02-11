# CREATED BY: Merlijn Sevenhuijsen  | merlijns@kth.se   | 200104073275
# CREATED BY: Hugo Westergård       | hugwes@kth.se     | 200011289659

import torch.nn as nn


class NeuralNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size, size_layers_hidden):
        super(NeuralNetwork, self).__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, size_layers_hidden)
        self.input_layer_activation = nn.ReLU()

        # Create second hidden layer with ReLU activation
        self.hidden_layer = nn.Linear(size_layers_hidden, size_layers_hidden)
        self.hidden_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(size_layers_hidden, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute input layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute second hidden layer
        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out