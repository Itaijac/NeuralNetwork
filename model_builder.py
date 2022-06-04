import numpy as np
import os.path
from keras.datasets import mnist
from keras.utils import np_utils
import cv2

from network import Network
from layer import Layer
from activations import sigmoid, sigmoid_prime

MODEL_PATH = r'E:\Machine Learning\Convolutional Neural Network\model'


def prepare_image(file_path, network):
    """
    This function prepares the image for the Neural Network, by centering the digit
    and resizing it.
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Finding contours and cropping the image to the bounding box
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]

    # Making sure the cropped image is a square, if not, padding it
    if image.shape[0] > image.shape[1]:
        diff = (image.shape[0]-image.shape[1])/2
        left = int(np.ceil(diff))
        right = int(np.floor(diff))
        image = cv2.copyMakeBorder(
            image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif image.shape[1] > image.shape[0]:
        diff = (image.shape[1]-image.shape[0])/2
        top = int(np.ceil(diff))
        bottom = int(np.floor(diff))
        image = cv2.copyMakeBorder(
            image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Resizing the image and padding it
    image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    image = cv2.bitwise_not(image)

    # Preparing the image for the Neural Network
    image = image.reshape(1, 28 * 28)
    image = image.astype('float32')
    image /= 255.0

    # Remove this line if you don't want to see the filters
    # network.show_filter(image)

    return image


def test(network):
    """
    This function tests the Neural Network by showcasing a bunch
    of digits to it (10,000 images) and seeing how accurate the
    Neural Network is.
    """

    # Loading the data
    (x_test, y_test) = mnist.load_data()[1]
    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
    x_test = x_test.astype('float32')
    x_test /= 255.0

    # Testing the model with the data
    ctr = 0
    test_amount = len(x_test)
    for i in range(test_amount):
        # Getting the prediction of the image
        val = network.predict(x_test[i])[0]
        # Checking if the prediction is correct
        if val == y_test[i]:
            ctr += 1

    # Printing the accuracy
    accuracy = ctr / test_amount * 100
    print(f'Got {ctr} right out of {test_amount}. Accuracy is {accuracy}')


def save(network):
    """
    This function saves the Neural Network to a file.
    """
    network.save(MODEL_PATH)


def load(network):
    """
    This function loads the Neural Network from a file.
    """
    network.load(MODEL_PATH)


def train(network):
    """
    This function trains the Neural Network by showcasing a bunch
    of digits to it (60,000 images) over and over until the Neural Network
    will be able to tell which digit is which by itself.
    """
    # Loading the data
    (x_train, y_train) = mnist.load_data()[0]
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255.0
    y_train = np_utils.to_categorical(y_train)

    # Training the model with the data
    network.train(x_train, y_train, epoch=15)


def launch(file_path):
    """
    This generator launches the Neural Network.
    """
    # Building the Neural Network
    network = Network()
    network.add(Layer(28 * 28, 16, sigmoid, sigmoid_prime))
    network.add(Layer(16, 16, sigmoid, sigmoid_prime))
    network.add(Layer(16, 10, sigmoid, sigmoid_prime))

    # Either loading or training the model
    if os.path.isdir(MODEL_PATH):
        load(network)
    else:
        train(network)

    # Getting the prediction of the image
    while True:
        image = prepare_image(file_path, network)
        prediction, coinfidence, probabilities = network.predict(image)
        yield prediction, coinfidence, probabilities
