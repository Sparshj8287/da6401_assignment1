from typing import List, Tuple

import numpy as np


from ..layers.dense import Dense


class NesterovGradientDescent:
    def __init__(self, momentum: float, **_) -> None:
        """
        Initialize Nesterov Accelerated Gradient (NAG) optimizer.

        Parameters:
        - momentum (float): Momentum factor (typically between 0.5 and 0.99).
        """

        self.momentum = momentum
        self.prev_uw_list: List[np.ndarray] = list()
        self.prev_ub_list: List[np.ndarray] = list()


    def add_params(self, w_shape: Tuple[int, int], b_shape: Tuple[int, int]):

        """
        Initialize Nesterov Accelerated Gradient (NAG) optimizer.

        Parameters:
        - momentum (float): Momentum factor (typically between 0.5 and 0.99).
        """

        self.prev_uw_list.append(np.zeros(w_shape))
        self.prev_ub_list.append(np.zeros(b_shape))

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
        # for output layer
        """
        Perform backward propagation using Nesterov Accelerated Gradient.

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

        out_batch = self.__get_out_batch(layers, x_train_batch, len(layers) - 1)



        dL_dz_batch = loss.gradient(out_batch[:, 1, :], y_train_batch)

        batch_size = len(dL_dz_batch)

        w_update_list = list()
        b_update_list = list()

        # calc output and hidden layer gradients
        for layer in range(len(layers) - 1, 0, -1):
            prev_layer = layers[layer - 1]
            prev_a = np.asarray(prev_layer.a)

            prev_a= prev_a.squeeze(0)

            w_update, b_update = self.__get_param_updates(
                dL_dz_batch, prev_a, layer, batch_size, lr
            )
            
            

            
            w_update_list.append(w_update)
            b_update_list.append(b_update)
            out_batch = self.__get_out_batch(layers, x_train_batch, layer - 1)



            dL_dz_batch = np.matmul(
                dL_dz_batch, layers[layer].w.T
            ) * prev_layer.activation.gradient(out_batch[:, 0, :])



        w_update, b_update = self.__get_param_updates(
            dL_dz_batch, x_train_batch, 0, batch_size, lr
        )




        w_update_list.append(w_update)
        b_update_list.append(b_update)

        return (w_update_list, b_update_list)

    def __get_param_updates(
        self,
        dL_dz_batch: np.ndarray,
        a: np.ndarray,
        layer: int,
        batch_size: int,
        lr: float,
    ) -> Tuple[float, float]:
        """
        Perform backward propagation using Nesterov Accelerated Gradient.

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

        dL_dw = np.matmul(a.T,dL_dz_batch) / batch_size
        dL_db = np.sum(dL_dz_batch, axis=0) / batch_size


        self.prev_uw_list[layer] = self.momentum * self.prev_uw_list[layer] + (lr * dL_dw)
        self.prev_ub_list[layer] = self.momentum * self.prev_ub_list[layer] + (lr * dL_db)


        return (self.prev_uw_list[layer], self.prev_ub_list[layer])

    def __get_out_batch(
        self, layers: List, x_train_batch: np.ndarray, layer: int
    ) -> np.ndarray:

        """
      Compute "look-ahead" activations for Nesterov momentum.

      Parameters:
      - layers(List[Dense]): List of layers in the neural network.
      - x_train_batch(np.ndarray): Input features for the batch.
      - end_layer_idx(int): Index of the last layer to compute activations.

      Returns:
      np.ndarray: Look-ahead activations up to the specified layer index.
      """

        uw = self.momentum * self.prev_uw_list[layer]
        ub = self.momentum * self.prev_ub_list[layer]


        prev_a = x_train_batch if layer == 0 else np.asarray(layers[layer - 1].a).squeeze(0)






        out_batch = np.asarray(Dense._forward(
            layers[layer].w - uw,  
            prev_a,                
            layers[layer].b - ub,  
            layers[layer].activation,
        )).transpose(1, 0, 2) 




        return out_batch


    @staticmethod
    def _optimizer_info()->None:
        """
         Harmless helper function to print optimizer info.

         Returns:
          None
        """

        print("[INFO] Using Nesterov Accelerated Gradient optimizer.")

    def __str__(self) -> str:
        return "NesterovGradientDescent"