from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.layers import Dropout
from si.neural_networks.optimizers import Optimizer
class TestDropoutLayer(TestCase):

    def setUp(self):
        """
        Initializes input data and other parameters required for the tests.
        """
        self.input_data = np.random.rand(32, 64)  # Example: matrix with 32 examples and 64 features
        self.probability = 0.3  # Dropout probability

    def test_forward_propagation_training(self):
        """
        Tests the forward_propagation method in training mode.
        """
        dropout_layer = Dropout(probability=self.probability)
        output = dropout_layer.forward_propagation(self.input_data, training=True)

        # Checks if the mask was correctly created
        self.assertIsNotNone(dropout_layer.mask, "The mask was not initialized correctly.")

        # Checks if the output has the same shape as the input
        self.assertEqual(output.shape, self.input_data.shape, "The output shape does not match the input shape.")

        # Checks if scaling was correctly applied
        scaling_factor = 1 / (1 - self.probability)
        scaled_input = self.input_data * dropout_layer.mask * scaling_factor
        np.testing.assert_array_almost_equal(output, scaled_input, err_msg="The output was not correctly scaled.")

    def test_forward_propagation_inference(self):
        """
        Tests the forward_propagation method in inference mode.
        """
        dropout_layer = Dropout(probability=self.probability)
        output = dropout_layer.forward_propagation(self.input_data, training=False)

        # Checks if the output is equal to the input (no modifications)
        np.testing.assert_array_equal(output, self.input_data, "The output in inference mode does not match the input.")

    def test_backward_propagation(self):
        """
        Tests the backward_propagation method.
        """
        dropout_layer = Dropout(probability=self.probability)
        # Performs forward propagation in training mode to initialize the mask
        dropout_layer.forward_propagation(self.input_data, training=True)

        # Generates a simulated error for backward propagation
        output_error = np.random.rand(32, 64)
        input_error = dropout_layer.backward_propagation(output_error)

        # Checks if input_error has the same shape as output_error
        self.assertEqual(input_error.shape, output_error.shape, "The shape of the propagated error does not match the expected shape.")

        # Checks if the error was correctly adjusted by the mask
        np.testing.assert_array_almost_equal(
            input_error,
            output_error * dropout_layer.mask,
            err_msg="Error in multiplying the error by the mask."
        )

    def test_output_shape(self):
        """
        Tests the output_shape method.
        """
        dropout_layer = Dropout(probability=self.probability)
        # Performs forward propagation to initialize the input shape
        dropout_layer.forward_propagation(self.input_data, training=True)

        # Checks if the output_shape matches the input_shape
        self.assertEqual(dropout_layer.output_shape(), self.input_data.shape, "The output shape does not match the input shape.")

    def test_parameters(self):
        """
        Tests the parameters method.
        """
        dropout_layer = Dropout(probability=self.probability)
        # Checks if the layer returns 0 trainable parameters
        self.assertEqual(dropout_layer.parameters(), 0, "The Dropout layer should have 0 trainable parameters.")