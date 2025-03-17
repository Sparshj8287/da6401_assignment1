import numpy as np
from ..weight_init import WeightInit
from ..activations.relu import ReLU
from typing import List
import unittest

class Dense:
    def __init__(self,
                 neurons: int,
                 activation: str,
                 weight_init=WeightInit.random,
                 w: np.ndarray = None,
                 b: np.ndarray = None) -> None:
        """
        Initialize Dense layer parameters.

        Parameters:
        - neurons (int): Number of neurons in the layer.
        - activation (str): Activation function to apply.
        - weight_init (callable): Method to initialize weights.
        - w (np.ndarray): Optional weight matrix. If None, initialized later.
        - b (np.ndarray): Optional bias vector. If None, initialized randomly.
        """

        self.neurons = neurons
        self.activation = activation
        self.weight_init = weight_init

        # Initialize weights if provided, else set empty array for later initialization
        self.w = w if w is not None else np.array([])

        # Initialize biases randomly if not provided
        self.b = b if b is not None else np.random.randn(1, neurons)

        # Containers to store intermediate values during backpropagation
        self.a = []
        self.z = []

    def forward(self, x: np.ndarray, backprop: bool) -> np.ndarray:
        """
        Perform forward propagation through the Dense layer.

        Parameters:
        - x (np.ndarray): Input data array.
        - backprop (bool): Flag indicating whether to store intermediate values for backpropagation.

        Returns:
        - np.ndarray: Activation output from the current layer.
        """

        z, a = Dense._forward(self.w, x, self.b, self.activation)

        # Store intermediate activations if backpropagation is required
        if backprop:
            self.z.append(z)
            self.a.append(a)

        return a

    @staticmethod
    def _forward(w: np.ndarray, x: np.ndarray, b: np.ndarray, activation: str) -> List[np.ndarray]:
        """
        Compute linear transformation and apply activation function.

        Parameters:
        - w (np.ndarray): Weight matrix.
        - x (np.ndarray): Input data array.
        - b (np.ndarray): Bias vector.
        - activation (str): Activation function instance with 'activate' method.

        Returns:
        - List[np.ndarray]: Linear output (z) and activated output (a).
        """

        # Linear transformation: z = xW + b
        z = np.add(np.dot(x, w), b)

        # Apply activation function on linear output
        a = activation.activate(z)

        return [z, a]

    def set_w(self, input_dim) -> None:
        """
        Initialize weight matrix based on input dimension and number of neurons.

        Parameters:
        - input_dim (int): Dimension of input features.
        
        Returns:
         None
         """

         # Initialize weights using provided weight initialization method
        self.w = self.weight_init((input_dim, self.neurons))

    def __str__(self) -> str:
         """
         String representation of Dense layer weights and biases.

         Returns:
         - str: Formatted string showing weights and biases.
         """
         return f"Weights:\n{self.w}\nBiases:\n{self.b}"

    @staticmethod
    def _print_layer_info(neurons:int)->None:
        """
         Harmless helper function to print basic layer information. 
         This function does not affect model logic.

         Parameters:
         - neurons(int): Number of neurons in the layer.

         Returns:
          None
        """
        print(f"[INFO] Dense layer created with {neurons} neurons.")


class TestDense(unittest.TestCase):
    def setUp(self):
        self.neurons = 4
        self.activation = ReLU()
        self.dense = Dense(neurons=self.neurons, activation=self.activation)
        self.x = np.random.randn(3, 5) 
        self.dense.set_w(5)  
    
    def test_forward_shape(self):
        output = self.dense.forward(self.x, backprop=False)
        self.assertEqual(output.shape, (3, self.neurons))
    
    def test_set_w_shape(self):
        self.assertEqual(self.dense.w.shape, (5, self.neurons))
        self.assertEqual(self.dense.b.shape, (1, self.neurons))
    
if __name__ == "__main__":
    unittest.main()
