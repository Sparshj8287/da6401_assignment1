from typing import List, Tuple

import numpy as np


class NAdam:
    def __init__(self, beta1: float, beta2: float, epsilon: float, **_) -> None:
        """
        Initialize NAdam optimizer parameters.

        Parameters:
        - beta1 (float): Exponential decay rate for first moment estimates.
        - beta2 (float): Exponential decay rate for second moment estimates.
        - epsilon (float): Small constant to prevent division by zero.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mw_list: List[np.ndarray] = list()
        self.mb_list: List[np.ndarray] = list()
        self.vw_list: List[np.ndarray] = list()
        self.vb_list: List[np.ndarray] = list()

    def add_params(self, w_shape: Tuple[int, int], b_shape: Tuple[int, int]):
        """
        Initialize moment estimates with zeros for a new layer.

        Parameters:
        - w_shape (Tuple[int, int]): Shape of weights.
        - b_shape (Tuple[int, int]): Shape of biases.
        """
        self.mw_list.append(np.zeros(w_shape))
        self.mb_list.append(np.zeros(b_shape))
        self.vw_list.append(np.zeros(w_shape))
        self.vb_list.append(np.zeros(b_shape))

    def backward(
        self,
        y_hat_batch: np.ndarray,
        x_train_batch: np.ndarray,
        y_train_batch: np.ndarray,
        layers: List,
        lr: float,
        loss: object,
        epoch: int,
    ) -> Tuple[np.ndarray]:
        """
        Perform backward propagation using NAdam optimizer.

        Parameters:
        - y_hat_batch (np.ndarray): Predicted outputs for the batch.
        - x_train_batch (np.ndarray): Input features for the batch.
        - y_train_batch (np.ndarray): True labels for the batch.
        - layers (List): List of layers in the neural network.
        - lr (float): Learning rate.
        - loss (object): Loss function object with gradient method.
        - epoch (int): Current epoch number.

        Returns:
         Tuple[List[np.ndarray],List[np.ndarray]]: Lists of weight and bias updates.
        """

        # for output layer
        dL_dz_batch = loss.gradient(y_hat_batch, y_train_batch)
        batch_size = len(dL_dz_batch)

        w_update_list = list()
        b_update_list = list()

        # calc output and hidden layer gradients
        for layer in range(len(layers) - 1, 0, -1):
            prev_layer = layers[layer - 1]
            prev_z = np.array(prev_layer.z)
            prev_z = prev_z.reshape(prev_z.shape[1], -1)
            prev_a = np.array(prev_layer.a)
            prev_a = prev_a.reshape(prev_a.shape[1], -1)

            w_update, b_update = self.__get_param_updates(
                dL_dz_batch, prev_a, layer, batch_size, lr, epoch
            )

            w_update_list.append(w_update.T)
            b_update_list.append(b_update)

            dL_dz_batch = np.matmul(
                dL_dz_batch, layers[layer].w.T
            ) * prev_layer.activation.gradient(prev_z)

        w_update, b_update = self.__get_param_updates(
            dL_dz_batch, x_train_batch, 0, batch_size, lr, epoch
        )

        w_update_list.append(w_update.T)
        b_update_list.append(b_update)

        return (w_update_list, b_update_list)

    def __get_param_updates(
        self,
        dL_dz_batch: np.ndarray,
        a: np.ndarray,
        layer: int,
        batch_size: int,
        lr: float,
        epoch: int,
    ) -> Tuple[float, float]:

        """
       Compute weight and bias updates using NAdam optimization.

       Parameters:
       - dL_dz_batch(np.ndarray): Gradient of loss w.r.t. pre-activation outputs.
       - a_prev(np.ndarray): Activations from previous layer.
       - layer(int): Index of current layer.
       - batch_size(int): Number of samples in the batch.
       - lr(float): Learning rate.
       - epoch(int): Current epoch number.

       Returns:
       Tuple[np.ndarray,np.ndarray]: Weight and bias updates.
       """
        dL_dw = np.matmul(dL_dz_batch.T, a) / batch_size
        dL_db = np.sum(dL_dz_batch, axis=0) / batch_size

        self.mw_list[layer] = (
            self.beta1 * self.mw_list[layer] + (1 - self.beta1) * dL_dw
        )
        self.mb_list[layer] = (
            self.beta1 * self.mb_list[layer] + (1 - self.beta1) * dL_db
        )

        self.vw_list[layer] = (
            self.beta2 * self.vw_list[layer] + (1 - self.beta2) * dL_dw**2
        )
        self.vb_list[layer] = (
            self.beta2 * self.vb_list[layer] + (1 - self.beta2) * dL_db**2
        )

        mw_hat = self.mw_list[layer] / (1 - np.power(self.beta1, epoch + 1))
        mb_hat = self.mb_list[layer] / (1 - np.power(self.beta1, epoch + 1))
        vw_hat = self.vw_list[layer] / (1 - np.power(self.beta2, epoch + 1))
        vb_hat = self.vb_list[layer] / (1 - np.power(self.beta2, epoch + 1))

        w_update = (lr / np.sqrt(vw_hat + self.epsilon)) * (
            self.beta1 * mw_hat
            + (1 - self.beta1) * dL_dw / (1 - self.beta1 ** (epoch + 1))
        )
        b_update = (lr / (np.sqrt(vb_hat + self.epsilon))) * (
            self.beta1 * mb_hat
            + (1 - self.beta1) * dL_db / (1 - self.beta1 ** (epoch + 1))
        )

        return (w_update, b_update)

    @staticmethod
    def _optimizer_info()->None:
        """
         Harmless helper function to print optimizer info.

         Returns:
          None
          """

        print("[INFO] Using NAdam optimizer.")

    def __str__(self) -> str:
        return "NAdam"