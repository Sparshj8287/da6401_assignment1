import unittest
import numpy as np

class Tanh:
    def activate(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the tanh activation function.

        Tanh function: (e^z - e^(-z)) / (e^z + e^(-z))
        Maps input values to the range (-1, 1).

        Parameters:
        - z (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output array after applying tanh.
        """
        # Compute positive and negative exponentials
        exp_pos = np.exp(z)
        exp_neg = np.exp(-z)
        
        # Tanh formula
        a = np.divide((exp_pos - exp_neg), (exp_pos + exp_neg))
        
        return a

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the tanh function.

        Gradient formula: 1 - tanh(z)^2

        Parameters:
        - z (np.ndarray): Input array.

        Returns:
        - np.ndarray: Gradient of the tanh function.
        """
        tanh_output = self.activate(z)
        grad = 1 - np.square(tanh_output)
        
        return grad

    @staticmethod
    def _info() -> None:
        """
        Harmless helper method to print activation function info.
        
        Returns:
        None
        """
        print("[INFO] Tanh activation function selected.")


class TestTanh(unittest.TestCase):
    def setUp(self):
        self.tanh = Tanh()
        self.z = np.array([[1.0, -2.0, 3.0], [0.0, -0.5, 2.0]])
    
    def test_activate(self):
        expected_activation = np.tanh(self.z)
        np.testing.assert_array_almost_equal(self.tanh.activate(self.z), expected_activation, decimal=6)
    
    def test_gradient(self):
        expected_grad = 1 - np.tanh(self.z) ** 2
        np.testing.assert_array_almost_equal(self.tanh.gradient(self.z), expected_grad, decimal=6)
    
if __name__ == "__main__":
    unittest.main()
