import numpy as np 
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import wandb
from typing import Tuple
from keras.utils import to_categorical
from keras.datasets import fashion_mnist, mnist

def get_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = {"fashion_mnist": fashion_mnist.load_data,
        "mnist": mnist.load_data}[dataset_name]
    (x_train, y_train), (x_test, y_test) = d()

    # reshaping and normalizing
    x_train_reshaped = x_train.reshape(len(x_train), -1) / 255
    x_test_reshaped = x_test.reshape(len(x_test), -1) / 255

    # one-hot encoding
    y_train_reshaped = to_categorical(y_train)
    y_test_reshaped = to_categorical(y_test)

    return (x_train_reshaped, y_train_reshaped, x_test_reshaped, y_test_reshaped)

wandb.init(project='DA6401_Assignment1')

# Load the Fashion MNIST dataset
x_train, y_train, x_test, y_test = get_dataset('fashion_mnist')

print(x_train.shape)
print(y_train.shape)

d= np.c_[x_train, y_train]

print(d.shape)

# Plot one image from each class
class_mapping= ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_classes=10
img_list=[]
class_list=[]
for i in range(num_classes):
    position = np.argmax(y_train==i)
    image = x_train[position,:,:]
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(class_mapping[i])
    img_list.append(image)
    class_list.append(class_mapping[i])
    
wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(img_list, class_list)]})
