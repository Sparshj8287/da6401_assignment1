from typing import List, Tuple

import numpy as np

class BatchGradientDescent:
    def __init__(self, **_) -> None:
        # No special initialization required for batch gradient descent

        super().__init__()

    def add_params(self, w_shape: Tuple[int, int], b_shape: Tuple[int, int]):
        # Batch Gradient Descent doesn't require additional parameter initialization
        pass

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
        Perform backward propagation using Batch Gradient Descent.

        Parameters:
        - y_hat_batch (np.ndarray): Predicted outputs for the batch.
        - x_train_batch (np.ndarray): Input features for the batch.
        - y_train_batch (np.ndarray): True labels for the batch.
        - layers (List): List of layers in the neural network.
        - lr (float): Learning rate.
        - loss (object): Loss function object with gradient method.
        - epoch (int): Current epoch number.

        Returns:
        - Tuple[List[np.ndarray], List[np.ndarray]]: Lists of weight and bias updates.
        """
        dL_dz_batch = loss.gradient(y_hat_batch, y_train_batch)



        batch_size = len(dL_dz_batch)

        w_update_list = list()
        b_update_list = list()

        # calc output and hidden layer gradients
        for layer in range(len(layers) - 1, 0, -1):
            prev_layer = layers[layer - 1]


            prev_z = np.array(prev_layer.z)
            prev_z= prev_z.reshape(prev_z.shape[1],-1)
            prev_a = np.array(prev_layer.a)
            prev_a= prev_a.reshape(prev_a.shape[1],-1)

            w_update, b_update = self.__get_param_updates(
                dL_dz_batch, prev_a, batch_size, lr
            )


            w_update_list.append(w_update.T)
            b_update_list.append(b_update)

            # print("dl_dz_batch", dL_dz_batch.shape)
            # print("w", layers[layer].w.shape)

            dL_dz_batch = np.matmul(
                dL_dz_batch, layers[layer].w.T
            ) * prev_layer.activation.gradient(prev_z)



        w_update, b_update = self.__get_param_updates(
            dL_dz_batch, x_train_batch, batch_size, lr
        )



        # print("w_update", w_update.shape)

        w_update_list.append(w_update.T)
        b_update_list.append(b_update)

        return (w_update_list, b_update_list)

    def __get_param_updates(
        self,
        dL_dz_batch: np.ndarray,
        a: np.ndarray,
        batch_size: int,
        lr: float,
    ) -> Tuple[float, float]:

        """
        Compute weight and bias updates using gradient descent.

        Parameters:
        - dL_dz_batch (np.ndarray): Gradient of loss w.r.t. pre-activation outputs.
        - a_prev (np.ndarray): Activations from previous layer.
        - batch_size (int): Number of samples in the batch.
        - lr (float): Learning rate.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Weight and bias updates.
        """
        dL_dw = np.matmul(dL_dz_batch.T, a) / batch_size
        dL_db = np.sum(dL_dz_batch, axis=0) / batch_size

        w_update = dL_dw * lr
        b_update = dL_db * lr

        return (w_update, b_update)

    @staticmethod
    def _optimizer_info() -> None:
        """
        Harmless helper function to print optimizer info.
        
        Returns:
         None
         """
        
        print("[INFO] Using Batch Gradient Descent optimizer.")

    def __str__(self):
        return "Batch Gradient Descent"