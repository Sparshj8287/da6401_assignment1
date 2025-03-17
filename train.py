import argparse
import numpy as np
import os
import pickle
import importlib
import uuid
from typing import Tuple
from tqdm import tqdm
import numpy as np
from sparshy import Model
from sparshy.activations import Identity, ReLU, Sigmoid, Softmax, Tanh
from sparshy.layers import Dense
from sparshy.loss import CrossEntropy, MeanSquaredError
from sparshy.optimizers import (
    BatchGradientDescent,
    MomentumGradientDescent,
    NesterovGradientDescent,
    RMSProp,
    Adam,
    NAdam


)
from sparshy.weight_init import WeightInit
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical

import wandb

# Activation functions dictionary
activation = {
    "identity": Identity,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "ReLU": ReLU,
    "tanh": Tanh,
}

# Optimizers dictionary
optimiser = {
    "sgd": BatchGradientDescent,
    "momentum": MomentumGradientDescent,
    "nag": NesterovGradientDescent,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": NAdam,
}
# Loss functions dictionary
loss = {"cross_entropy": CrossEntropy,
        "mean_squared_error": MeanSquaredError}



def get_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Load and preprocess the dataset.

    Parameters:
    - dataset (str): Name of the dataset (e.g., 'mnist' or 'fashion_mnist').

    Returns:
    - Tuple containing reshaped and normalized training and testing data.
    """



    d = importlib.import_module(f"keras.datasets.{dataset}")
    (x_train, y_train), (x_test, y_test) = d.load_data()

    x_train_reshaped = x_train.reshape(len(x_train), 784) / 255
    y_train_reshaped = np.zeros((len(y_train), 10))
    for sample in range(len(y_train)):
        y_train_reshaped[sample][y_train[sample]] = 1

    x_test_reshaped = x_test.reshape(len(x_test), 784) / 255
    y_test_reshaped = np.zeros((len(y_test), 10))
    for sample in range(len(y_test)):
        y_test_reshaped[sample][y_test[sample]] = 1

    return (x_train_reshaped, y_train_reshaped, x_test_reshaped, y_test_reshaped)

def build_model(num_layers:int,
                hidden_size:int,
                activation_func:str,
                weight_init:str) -> Model:
    
    """
    Build a neural network model with specified parameters.

    Parameters:
    - num_layers (int): Number of hidden layers.
    - hidden_size (int): Number of neurons in each hidden layer.
    - activation_func (str): Activation function for hidden layers.
    - weight_init (str): Weight initialization method.

    Returns:
    - Model: A compiled neural network model.
    """

    layers = [
        Dense(hidden_size, activation[activation_func](), WeightInit[weight_init]),
    ]

    for i in range(num_layers-1):
        layers.append(Dense(hidden_size, activation[activation_func](), WeightInit[weight_init]))

    layers.append(Dense(10, activation["softmax"](), WeightInit[weight_init]))

    return Model(layers)


def train(args):

    """
    Train the neural network model with specified arguments.

    Parameters:
      args: Command-line arguments specifying training configuration.
      
      """
      
    if args.log_location == "wandb":
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"hl_{args.num_layers}_sz_{args.hidden_size}_bs_{args.batch_size}_opt_{args.optimizer}_act_{args.activation}_id_{wandb.run.id}"
        model_name = wandb.run.name
    else:
        model_name = str(uuid.uuid4())


    x_train, y_train, x_test, y_test = get_dataset(args.dataset)
    

    model = build_model(
        args.num_layers, args.hidden_size, args.activation, args.weight_init
    )

    params = {
        "momentum": args.momentum,
        "beta": args.beta,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "epsilon": args.epsilon,
    }

    model.compile(
        optimiser[args.optimizer](**params),
        loss[args.loss](),

    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        log_location=args.log_location,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    with open(f"{args.output_folder}/{model_name}.nn", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")

    # Define the groups for the arguments
    data_args = parser.add_argument_group('Data Args')
    model_args = parser.add_argument_group('Model Args')
    train_args = parser.add_argument_group('Training Args')
    other_args = parser.add_argument_group('Other Args')

    # Data arguments
    data_args.add_argument("-d", "--dataset", type=str, default="fashion_mnist",choices=["mnist","fashion_mnist"], help="Dataset to train on.")

    # Model arguments
    model_args.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers in the model.")
    model_args.add_argument("-sz", "--hidden_size", type=int, default=128, help="Size of hidden layers.")
    model_args.add_argument("-a", "--activation", type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "ReLU"],help="Activation function to use.")
    model_args.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "he"],help="Weight initialization strategy.")

    # Training arguments
    train_args.add_argument("-o", "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help="Optimizer to use.")
    train_args.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Loss function to use.")
    train_args.add_argument("-lr", "--learning_rate", type=float, default=0.016845152698391038, help="Learning rate.")
    train_args.add_argument("-m", "--momentum", type=float, default=0.4843657663446104, help="Momentum for optimizer.")
    train_args.add_argument("-beta", "--beta", type=float, default=0.7187842440442459, help="Beta for optimizer.")
    train_args.add_argument("-beta1", "--beta1", type=float, default=0.08043285377412115, help="Beta1 for optimizer.")
    train_args.add_argument("-beta2", "--beta2", type=float, default=0.7811457452810221, help="Beta2 for optimizer.")
    train_args.add_argument("-eps", "--epsilon", type=float, default=0.007948507751484618, help="Epsilon for optimizer.")
    train_args.add_argument("-ep", "--epochs", type=int, default=10, help="Number of epochs to train for.")
    train_args.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size.")
    train_args.add_argument("-w_d", "--weight_decay", type=float, default=0.0030704617860640273, help="Weight decay.")

    # Other arguments
    other_args.add_argument("-ll", "--log_location", type=str, default="wandb", help="Where to log results.")
    other_args.add_argument("-we", "--wandb_entity", type=str, default="sjshiva8287", help="Wandb entity.")
    other_args.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment1", help="Wandb project.")
    other_args.add_argument("-of", "--output_folder", type=str, default="models", help="Folder to save model.")

    args = parser.parse_args()

    print("Running with arguments:\n-------------")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("------------------------------------")

    train(args)
    print(" Training complete!")