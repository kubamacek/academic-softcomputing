import os
import numpy as np
import sklearn
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
import random

from tensorflow.python.keras.models import load_model


def save_model(model, model_name_to_save):
    model.save(os.path.join('models/', model_name_to_save))


def load_model_from_file(saved_model_name):
    model = load_model(os.path.join('models/', saved_model_name))
    print("loaded model:", saved_model_name)
    model.summary()
    return model


def get_number_of_classes(train_data_directory, train_directory='Train'):
    """
    Each type of traffic sign has its own directory in train data path.
    Function returns number of these directories.
    """
    return len(os.listdir(os.path.join(train_data_directory, train_directory)))


def prepare_train_data(train_data_directory):
    """
    Function parses images from train data directory and returns two numpy arrays with all images and their labels.
    """
    train_data = list()
    labels = list()
    for traffic_sign_class in range(get_number_of_classes(train_data_directory)):
        traffic_sign_path = os.path.join(train_data_directory, 'Train', str(traffic_sign_class))
        traffic_sign_images = os.listdir(traffic_sign_path)
        for image_path in traffic_sign_images:
            image = Image.open(os.path.join(traffic_sign_path, image_path))
            image = image.convert('L')
            image = image.resize((30, 30))
            image = np.array(image)
            train_data.append(image)
            labels.append(traffic_sign_class)
    data = np.array(train_data)
    labels = np.array(labels)
    data.shape
    labels.shape

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=7)
    train_labels = to_categorical(train_labels, get_number_of_classes('data'))
    test_labels = to_categorical(test_labels, get_number_of_classes('data'))

    return train_data, test_data, train_labels, test_labels


def prepare_sample_data(sample_data_directory='Sample_data'):
    sample_data = list()
    labels = list()
    for traffic_sign_class in range(get_number_of_classes(sample_data_directory, "")):
        traffic_sign_path = os.path.join(sample_data_directory, str(traffic_sign_class))
        traffic_sign_images = os.listdir(traffic_sign_path)

        for i in range(0, 5):
            image_path = random.choice(traffic_sign_images)
            image = Image.open(os.path.join(traffic_sign_path, image_path))
            image = image.convert('L')
            image = image.resize((30, 30))
            image = np.array(image)
            sample_data.append(image)
            labels.append(traffic_sign_class)

    data = np.array(sample_data)
    labels = np.array(labels)
    data.shape
    labels.shape
    labels = to_categorical(labels, get_number_of_classes('data'))
    return data, labels


def test_model(model_name_to_test):
    sample_data, sample_labels = prepare_sample_data()
    model = load_model_from_file(model_name_to_test)
    print("Generate predictions for few samples:")
    success = 0
    for sample_id in range(1, len(sample_data)):
        predictions = model.predict(sample_data[sample_id - 1:sample_id])
        acc = sklearn.metrics.accuracy_score(sample_labels[sample_id - 1], predictions[0].round())
        if acc == 1.0:
            success += 1
    accuracy = (success/len(sample_data))*100
    print(model_name_to_test, "acc: ", accuracy, "tests succeeded/failed: {}/{}".format(success, len(sample_data)-success))


# def createSequentialModel_core():
#     model = Sequential()
#     model.add(Dense(30, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(8))
#     model.add(Dense(get_number_of_classes('data'), activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, batch_size=32, epochs=16, validation_data=(X_test, y_test))
#     model.summary()


def createSequentialModel(nodes_layer_1, activation_func_1, nodes_layer_2, activation_func_2, loss_function, optimizer,
                          batch_size, epochs, model_name):
    model = Sequential()
    model.add(Dense(nodes_layer_1, activation=activation_func_1))
    model.add(Flatten())
    model.add(Dense(nodes_layer_2))
    model.add(Dense(get_number_of_classes('data'), activation=activation_func_2))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.summary()
    save_model(model, model_name)

def createSequentialModel_enhanced(nodes_layer_1, activation_func_1, nodes_layer_2, activation_func_2, nodes_layer_3,
                                   activation_func_3, loss_function, optimizer, batch_size, epochs, model_name):
    model = Sequential()
    model.add(Dense(nodes_layer_1, activation=activation_func_1))
    model.add(Flatten())
    model.add(Dense(nodes_layer_2))
    model.add(Dense(nodes_layer_3), activation=activation_func_2)
    model.add(Dense(get_number_of_classes('data'), activation=activation_func_3))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.summary()
    save_model(model, model_name)



def checkAccurancy(model_name_to_verify, example_data, example_labels, batch_size):
    model = load_model_from_file(model_name_to_verify)
    results = model.evaluate(example_data, example_labels, batch_size=batch_size)
    print(model_name_to_verify, "test_loss, test acc:", results)


# createSequentialModel_core()
X_train, X_test, y_train, y_test = prepare_train_data('data')

# createSequentialModel(128, "relu", 64, "softmax", "categorical_crossentropy", "adam", 32, 30, "best_model")       96-97% acc
checkAccurancy("best_model", X_test, y_test, 32)
test_model("best_model")





