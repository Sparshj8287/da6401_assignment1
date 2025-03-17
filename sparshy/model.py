import numpy as np 
from .layers import Dense 
from tqdm import tqdm
from typing import List
import unittest
from .activations import Identity, Softmax, ReLU
from .loss import CrossEntropy
import wandb
from .optimizers import BatchGradientDescent
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, layers:List) -> None:
        """
        Initialize the model with a list of layers.

        Parameters:
        - layers (List): List of layer objects defining the model architecture.
        """
        self.layers = layers

    def forward(self, x:np.ndarray)-> np.ndarray:
        """
        Perform forward propagation through all layers.

        Parameters:
        - x (np.ndarray): Input data.

        Returns:
        - np.ndarray: Output after forward propagation through all layers.
        """

        for layer in self.layers:
            x = layer.forward(x, backprop=True)
            
        return x
    
    def compile(self,
                optimizer: object,
                loss: object)-> None:
        self.optimizer= optimizer
        self.loss= loss
        """
        Compile the model by setting the optimizer and loss function.

        Parameters:
        - optimizer (object): Optimizer object for weight updates.
        - loss (object): Loss function object for calculating loss.
        
        Returns:
         None
         """

        self.layers[0].set_w(784)
        if self.optimizer.__str__() == "NesterovGradientDescent":
            optimizer.add_params(self.layers[0].w.shape, self.layers[0].b.shape)
        else:
            optimizer.add_params(self.layers[0].w.T.shape, self.layers[0].b.shape)

        for i in range(1, len(self.layers)):
            self.layers[i].set_w(self.layers[i-1].neurons)
            if self.optimizer.__str__() == "NesterovGradientDescent":
                optimizer.add_params(self.layers[i].w.shape, self.layers[i].b.shape)
            else:
                optimizer.add_params(self.layers[i].w.T.shape, self.layers[i].b.shape)
    

    def predict(self, x: np.ndarray, backprop: bool = False):
        """
    Perform prediction by forward propagating through the model.

    Parameters:
    - x (np.ndarray): Input data for prediction.
    - backprop (bool): Whether to store intermediate values for backpropagation. Default is False.

    Returns:
    - np.ndarray: Predicted output after forward propagation.
    """
        a = x.copy()
        for layer in self.layers:
            a= layer.forward(a,backprop)

        return a
    
    def show(self):
        """
    Display details of all layers in the model.

    Returns:
    None
    """
        for layer in self.layers:
            print(layer)


    def update_weights(
            self,
            w_update_list: np.ndarray,
            b_update_list: np.ndarray,
            lr: float,
            weight_decay: float,
    ):
        """
    Update the weights and biases of all layers in the model.

    Parameters:
    - w_update_list (np.ndarray): List of weight updates for each layer.
    - b_update_list (np.ndarray): List of bias updates for each layer.
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay factor for regularization.

    Returns:
    None
    """
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            # Update weights with gradient, learning rate, and weight decay
            self.layers[layer_idx].w -= (
                w_update_list[len(self.layers) - 1 - layer_idx]
                + lr * weight_decay * self.layers[layer_idx].w
            )

            # Update biases with gradient and learning rate
            self.layers[layer_idx].b -= (
                b_update_list[len(self.layers) - 1 - layer_idx]
                + lr * weight_decay * self.layers[layer_idx].b
            )

            # Clear stored activations and pre-activations for the current layer
            self.layers[layer_idx].a = []
            self.layers[layer_idx].z = []


    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        log_location: str,
        patience: int = 5  # Number of epochs to wait for improvement before stopping
    ):

        """
    Train the model on the provided training data.

    Parameters:
    - x_train (np.ndarray): Training input data.
    - y_train (np.ndarray): Training labels (one-hot encoded).
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): Number of samples per training batch.
    - lr (float): Learning rate for weight updates.
    - weight_decay (float): Weight decay factor for regularization.
    - log_location (str): Location to log training progress ('wandb' or 'console').
    - patience (int, optional): Number of epochs to wait for improvement in validation accuracy before early stopping. Default is 5.

    Returns:
    None
    """
        # Split dataset into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

        indices = np.arange(len(x_train))

        best_val_acc = 0  # Track the best validation accuracy
        epochs_without_improvement = 0  # Counter for early stopping

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            np.random.shuffle(indices)  # Shuffle indices for each epoch
            train_loss = 0
            train_acc = 0

            # Add a progress bar for the batch loop
            with tqdm(total=len(x_train), desc=f"Epoch {epoch}", unit="sample") as pbar:
                for i in range(0, len(x_train), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    x_train_batch = x_train[batch_indices]
                    y_train_batch = y_train[batch_indices]

                    # Forward pass and backward pass
                    y_hat_batch = self.predict(x_train_batch, True)
                    w_update_list, b_update_list = self.optimizer.backward(
                        y_hat_batch,
                        x_train_batch,
                        y_train_batch,
                        self.layers,
                        lr,
                        self.loss,
                        epoch,
                    )
                    self.update_weights(w_update_list, b_update_list, lr, weight_decay)

                    # Calculate loss and accuracy for the current batch
                    curr_train_loss = self.loss.calc_loss(y_hat_batch, y_train_batch)
                    train_loss += curr_train_loss * len(batch_indices)
                    train_acc += np.sum(np.argmax(y_hat_batch, axis=1) == np.argmax(y_train_batch, axis=1))

                    # Update the progress bar
                    pbar.update(len(batch_indices))
                    pbar.set_postfix(loss=curr_train_loss)

            # Compute average loss and accuracy for logging
            train_loss /= len(x_train)
            train_acc /= len(x_train)

            val_loss = self.get_loss(x_val, y_val)
            val_acc = self.get_accuracy(x_val, y_val)

            log_data = {
                "train": {
                    "loss": train_loss,
                    "acc": train_acc,
                },
                "val": {
                    "loss": val_loss,
                    "acc": val_acc,
                },
            }

            # Log data based on the specified log location
            if log_location == "wandb":
                wandb.log(log_data, step=epoch, commit=True)
            else:
                print(log_data)

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0  # Reset counter if validation accuracy improves
                print(f"Validation accuracy improved to {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement in validation accuracy for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. No improvement in validation accuracy for {patience} consecutive epochs.")
                break

    def get_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the loss of the model predictions on given data.

        Parameters:
        - x (np.ndarray): Input data for prediction.
        - y (np.ndarray): True labels corresponding to input data.

        Returns:
        - float: Computed loss value.
        """
        # Generate predictions for input data
        y_pred = self.predict(x)

        # Calculate loss between predicted outputs and true labels
        return self.loss.calc_loss(y_pred, y)


    def get_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model predictions on given data.

        Parameters:
        - x (np.ndarray): Input data for prediction.
        - y (np.ndarray): True labels corresponding to input data.

        Returns:
        - float: Accuracy value (between 0 and 1).
        """
        correct = 0

        # Iterate over all samples to compute accuracy
        for idx, sample in enumerate(x):
            # Predict output for each individual sample
            y_hat = self.predict(sample)

            # Check if predicted class matches true class label
            if np.argmax(y_hat) == np.argmax(y[idx]):
                correct += 1

        # Calculate accuracy as ratio of correct predictions to total samples
        return correct / len(x)



class TestForward(unittest.TestCase):
    def test_forward(self):
        model = Model(
            [
                Dense(128, Identity()),
                Dense(64, Identity()),
                Dense(10, Softmax()),
            ]
        )

        model.compile(
            optimizer=BatchGradientDescent(),
            loss=CrossEntropy(),
        
        )

        x = np.random.rand(100, 784)
        y = model.predict(x)
        self.assertEqual(y.shape, (100, 10))


if __name__ == "__main__":
    unittest.main()