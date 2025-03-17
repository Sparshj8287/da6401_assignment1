import numpy as np
import unittest

class MeanSquaredError:
    def calc_loss(self, y_hat: np.ndarray, y_train: np.ndarray) -> float:
        """
        Calculate the mean squared error (MSE) loss.

        Formula: 0.5 * mean((y_train - y_hat)^2)
        The factor of 0.5 is used to simplify gradient calculations.

        Parameters:
        - y_hat (np.ndarray): Predicted values.
        - y_train (np.ndarray): True target values.

        Returns:
        - float: Computed MSE loss.
        """
        mse_loss = 0.5 * np.mean(np.square(y_train - y_hat))
        return mse_loss

    def gradient(self, y_hat: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the MSE loss with respect to predictions.

        Formula: (y_hat - y_train) * y_hat * (1 - y_hat) / len(y_train)
        The gradient is scaled by the length of the training data.

        Parameters:
        - y_hat (np.ndarray): Predicted values.
        - y_train (np.ndarray): True target values.

        Returns:
        - np.ndarray: Gradient of the MSE loss.
        """
        grad = np.multiply((y_hat - y_train), y_hat * (1 - y_hat)) / len(y_train)
        return grad

    @staticmethod
    def _info() -> None:
        """
        Harmless helper method to print loss function info.
        
        Returns:
        None
        """
        print("[INFO] MeanSquaredError loss function selected.")

class TestMeanSquaredError(unittest.TestCase):
    def setUp(self):
        self.mse = MeanSquaredError()
        self.y_train = np.array([1.0, 0.0, 1.0, 0.0])
        self.y_hat = np.array([0.9, 0.1, 0.8, 0.2])

    def test_calc_loss(self):
        expected_loss = 0.5 * np.sum((self.y_train - self.y_hat) ** 2) / len(self.y_train)
        self.assertAlmostEqual(self.mse.calc_loss(self.y_hat, self.y_train), expected_loss, places=6)

    def test_gradient(self):
        expected_grad = (self.y_hat - self.y_train) * self.y_hat * (1 - self.y_hat) / len(self.y_train)
        np.testing.assert_almost_equal(self.mse.gradient(self.y_hat, self.y_train), expected_grad, decimal=6)

if __name__ == "__main__":
    unittest.main()
