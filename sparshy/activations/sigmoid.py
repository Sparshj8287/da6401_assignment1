import unittest
import numpy as np

class Sigmoid:
    def activate(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function.

        Sigmoid function: 1 / (1 + e^(-z))
        Maps input values to the range (0, 1).

        Parameters:
        - z (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output after applying the sigmoid function.
        """
        a = np.divide(1, 1 + np.exp(-z))  
        return a

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the sigmoid function.

        Gradient formula: sigmoid(z) * (1 - sigmoid(z))

        Parameters:
        - z (np.ndarray): Input array.

        Returns:
        - np.ndarray: Gradient of the sigmoid function.
        """
        sig = self.activate(z)
        grad = sig * (1 - sig)
        return grad

    @staticmethod
    def _info() -> None:
        """
        Harmless helper method to print activation function info.
        
        Returns:
        None
        """
        print("[INFO] Sigmoid activation function selected.")


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()
        self.z = np.array([[1.0, -2.0, 3.0], [0.0, -0.5, 2.0]])
    
    def test_activate(self):
        expected_activation = 1 / (1 + np.exp(-self.z))
        np.testing.assert_array_almost_equal(self.sigmoid.activate(self.z), expected_activation, decimal=6)
    
    def test_gradient(self):
        expected_activation = 1 / (1 + np.exp(-self.z))
        expected_grad = expected_activation * (1 - expected_activation)
        np.testing.assert_array_almost_equal(self.sigmoid.gradient(self.z), expected_grad, decimal=6)
    
if __name__ == "__main__":
    unittest.main()