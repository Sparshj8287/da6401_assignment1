import unittest
import numpy as np

class Softmax:
    def activate(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the softmax activation function.

        Softmax function: Converts raw scores into probabilities that sum to 1.
        Formula: exp(z) / sum(exp(z)), applied along the last axis.

        Parameters:
        - z (np.ndarray): Input array (logits).

        Returns:
        - np.ndarray: Output array with probabilities.
        """

        shifted_z = z - np.max(z, axis=-1, keepdims=True)
        exp_values = np.exp(shifted_z)
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        
        return probabilities

    @staticmethod
    def _info() -> None:
        """
        Harmless helper method to print activation function info.
        
        Returns:
        None
        """
        print("[INFO] Softmax activation function selected.")


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = Softmax()
        self.z = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    
    def test_activate(self):
        exp_z = np.exp(self.z - np.max(self.z, axis=-1, keepdims=True))
        expected_activation = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(self.softmax.activate(self.z), expected_activation, decimal=6)
    
if __name__ == "__main__":
    unittest.main()
