import unittest
import numpy as np

class ReLU:
    def activate(self, z: np.ndarray) -> np.ndarray:
        # Apply ReLU activation: returns input if positive; otherwise returns zero
        a = np.where(z > 0, z, 0)
        return a

    def gradient(self, z: np.ndarray) -> np.ndarray:
        # Gradient of ReLU: 1 if input positive; otherwise 0
        grad = np.zeros(z.shape)
        grad[z > 0] = 1
        return grad

    @staticmethod
    def _info() -> None:
        print("[INFO] ReLU activation function selected.")

        
class TestReLU(unittest.TestCase):
    def setUp(self):
        self.relu = ReLU()
        self.z = np.array([[1.0, -2.0, 3.0], [0.0, -0.5, 2.0]])
    
    def test_activate(self):
        expected_activation = np.maximum(0, self.z)
        np.testing.assert_array_equal(self.relu.activate(self.z), expected_activation)
    
    def test_gradient(self):
        expected_grad = (self.z > 0).astype(int)
        np.testing.assert_array_equal(self.relu.gradient(self.z), expected_grad)
    
if __name__ == "__main__":
    unittest.main()
