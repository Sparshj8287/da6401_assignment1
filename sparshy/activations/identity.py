import numpy as np
import unittest

class Identity:
    def activate(self, z: np.ndarray) -> np.ndarray:
        # Identity activation returns input as output without any changes
        return np.copy(z)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        # Gradient of identity function is always 1
        return np.ones(z.shape)

    @staticmethod
    def _info() -> None:
        # Harmless helper method to print activation function info
        print("[INFO] Identity activation function selected.")


class TestIdentity(unittest.TestCase):
    def setUp(self):
        self.identity = Identity()
        self.z = np.array([[1.0, -2.0, 3.0], [0.5, -0.5, 2.0]])
    
    def test_activate(self):
        np.testing.assert_array_equal(self.identity.activate(self.z), self.z)
    
    def test_gradient(self):
        expected_grad = np.ones_like(self.z)
        np.testing.assert_array_equal(self.identity.gradient(self.z), expected_grad)
    
if __name__ == "__main__":
    unittest.main()
