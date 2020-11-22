from MLP import *
import itertools

# list_nodes_layer_1 = [512, 256, 128, 64]
list_nodes_layer_1 = [128]
list_activation_func_1 = ["softsign", "tanh", "selu", "elu", "exponential"] #["relu", "sigmoid", "softmax", "softplus",
list_nodes_layer_2 = [64]
list_activation_func_2 = ["relu", "sigmoid", "softmax",
                          "softplus"]  # , "softsigns", "tanh", "selu", "elu", "exponential"]
list_nodes_layer_3 = []
list_activation_func_3 = ["relu", "sigmoid", "softmax",
                          "softplus"]  # , "softsigns", "tanh", "selu", "elu", "exponential"]
list_loss_function = ["categorical_crossentropy"]
# , "sparse_categorical_crossentropy", "poisson", "binary_crossentropy",
#                   "kl_divergence", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
#                   "MeanSquaredLogarithmicError", "cosine_similarity"]
list_optimizer = ["SGD", "Adam", "RMSprop"]  # , "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
list_batch_size = [256]
list_epochs = [8]

custom_iterator = itertools.product(list_nodes_layer_1, list_activation_func_1, list_nodes_layer_2,
                                    list_activation_func_2, list_loss_function, list_optimizer, list_batch_size,
                                    list_epochs)


counter = 48
for config in custom_iterator:
    counter += 1
    print(counter, config)
    print("---------------------------------------------------------------------------------------------------")
    createSequentialModel(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7],
                          "temp_model_{}".format(counter))





# createSequentialModel(30, "relu", 8, "softmax", "categorical_crossentropy", "adam", 16, 16)
# createSequentialModel(30, "relu", 8, "softmax", "categorical_crossentropy", "adam", 64, 16)
# createSequentialModel(30, "relu", 8, "softmax", "categorical_crossentropy", "SGD", 64, 16)
# createSequentialModel(30, "relu", 8, "softmax", "mean_squared_error", "adam", 64, 16)
# createSequentialModel(30, "relu", 8, "relu", "mean_squared_error", "adam", 64, 16)
# createSequentialModel(30, "relu", 8, "softmax", "mean_squared_error", "SGD", 64, 16)
# createSequentialModel(30, "relu", 8, "softmax", "categorical_crossentropy", "SGD", 16, 16)
# createSequentialModel(30, "relu", 8, "softmax", "mean_squared_error", "adam", 16, 16)
# createSequentialModel(30, "relu", 8, "relu", "mean_squared_error", "adam", 16, 16)
# createSequentialModel(30, "relu", 8, "softmax", "mean_squared_error", "SGD", 16, 16)
