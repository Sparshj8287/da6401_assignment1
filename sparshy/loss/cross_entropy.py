import unittest
import numpy as np

class CrossEntropy:
    def calc_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the cross-entropy loss.

        Formula: -mean(y_true * log(y_pred))
        A small constant (1e-10) is added to y_pred for numerical stability.

        Parameters:
        - y_pred (np.ndarray): Predicted probabilities (output of softmax).
        - y_true (np.ndarray): True labels (one-hot encoded).

        Returns:
        - float: Computed cross-entropy loss.
        """
        epsilon = 1e-10  # Small constant to prevent log(0)
        loss = -np.mean(np.multiply(y_true, np.log(y_pred + epsilon)))
        return loss

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the cross-entropy loss with respect to predictions.

        Formula: gradient = y_pred - y_true

        Parameters:
        - y_pred (np.ndarray): Predicted probabilities (output of softmax).
        - y_true (np.ndarray): True labels (one-hot encoded).

        Returns:
        - np.ndarray: Gradient of the cross-entropy loss.
        """
        grad = np.subtract(y_pred, y_true)
        return grad

    @staticmethod
    def _info() -> None:
        """
        Harmless helper method to print loss function info.
        
        Returns:
        None
        """
        print("[INFO] CrossEntropy loss function selected.")


class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.ce = CrossEntropy()
        self.y_true = np.array([1, 0, 1, 0])
        self.y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    
    def test_calc_loss(self):
        expected_loss = -np.mean(self.y_true * np.log(self.y_pred + 1e-10))
        self.assertAlmostEqual(self.ce.calc_loss(self.y_pred, self.y_true), expected_loss, places=6)
    
    def test_gradient(self):
        expected_grad = self.y_pred - self.y_true
        np.testing.assert_almost_equal(self.ce.gradient(self.y_pred, self.y_true), expected_grad, decimal=6)

if __name__ == "__main__":
    unittest.main()
