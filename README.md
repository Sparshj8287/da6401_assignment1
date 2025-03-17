# da6401_assignment1
This repository provides tools to create and train perceptron-based neural networks. It includes a Python script ```(train.py)``` that allows users to conveniently train neural networks from the command line, with optional integration for experiment tracking via Weights & Biases.

# Github and Wandb Link
Github Repository Link: https://github.com/Sparshj8287/da6401_assignment1<br>
Wandb Report Link: https://wandb.ai/sjshiva8287/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTYwMDY4OQ

## Folder Structure

```
da6401_assignment1/
├── config.yml
├── question1.py
├── README.md
├── requirements.txt
├── sparshy/
│   ├── __init__.py
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── identity.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── softmax.py
│   │   └── tanh.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── dense.py
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── cross_entropy.py
│   │   └── mean_squared_error.py
│   ├── model.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── adam.py
│   │   ├── batch_gradient_descent.py
│   │   ├── momentum_gradient_descent.py
│   │   ├── nadam.py
│   │   ├── nesterov_gradient_descent.py
│   │   └── rmsprop.py
│   └── weight_init.py
└── train.py
```

## Usage

To train a neural network with this script, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Sparshj8287/da6401_assignment1.git
```

2. Navigate to the cloned directory:
```bash
cd da6401_assignment1
```

3. Install the required dependencies. It is recommended to use a virtual environment:
```bash
pip install -r requirements.txt
```

4. Run the training script train.py with the desired command-line arguments. Here's an example command:
```bash
python3 train.py --wandb_entity your_entity --wandb_project your_project -d fashion_mnist -ep 10 -bs 32 -l cross_entropy -o adam -lr 0.001 -nhl 2 -sz 64 -a ReLU
```

Replace myname and myprojectname with your actual Weights & Biases entity and project names.

## Command-line Arguments

The script supports the following command-line arguments:

- `-wp, --wandb_project`: Project name used to track experiments in Weights & Biases dashboard (default: 'myprojectname').
- `-we, --wandb_entity`: Wandb Entity used to track experiments in the Weights & Biases dashboard (default: 'myname').
- `-d, --dataset`: Dataset to use. Choices: ["mnist", "fashion_mnist"] (default: 'fashion_mnist').
- `-ep, --epochs`: Number of epochs to train neural network (default: 1).
- `-bs, --batch_size`: Batch size used to train neural network (default: 4).
- `-l, --loss`: Loss function to use. Choices: ["mean_squared_error", "cross_entropy"] (default: 'cross_entropy').
- `-o, --optimizer`: Optimizer to use. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] (default: 'sgd').
- `-lr, --learning_rate`: Learning rate used to optimize model parameters (default: 0.1).
- `-m, --momentum`: Momentum used by momentum and nag optimizers (default: 0.5).
- `-beta, --beta`: Beta used by rmsprop optimizer (default: 0.5).
- `-beta1, --beta1`: Beta1 used by adam and nadam optimizers (default: 0.5).
- `-beta2, --beta2`: Beta2 used by adam and nadam optimizers (default: 0.5).
- `-eps, --epsilon`: Epsilon used by optimizers (default: 0.000001).
- `-w_d, --weight_decay`: Weight decay used by optimizers (default: 0.0).
- `-w_i, --weight_init`: Weight initialization method. Choices: ["random", "Xavier"] (default: 'random').
- `-nhl, --num_layers`: Number of hidden layers in neural network (default: 1).
- `-sz, --hidden_size`: Number of hidden neurons in a feedforward layer (default: 4).
- `-a, --activation`: Activation function for hidden layers. Choices: ["identity", "sigmoid", "tanh", "ReLU"] (default: 'sigmoid').
- `-of, --output_folder`: Output folder for trained models (default: 'models').
- `-ll, --log_location`: Log location. Choices: ["wandb", "local"] (default: 'wandb').

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
