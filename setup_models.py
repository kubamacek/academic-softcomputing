from MLP import *
import itertools

# list_nodes_layer_1 = [512, 256, 128, 64]
list_nodes_layer_1 = [128]
list_activation_func_1 = ["relu", "sigmoid", "softmax", "softplus"]
# list_activation_func_1 = ["softsign", "tanh", "selu", "elu", "exponential"]
list_nodes_layer_2 = [64]
list_activation_func_2 = ["relu", "sigmoid", "softmax", "softplus"]  # , "softsigns", "tanh", "selu", "elu", "exponential"]
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
list_of_configs = list()
list_of_acc = list()
for config in custom_iterator:
    counter += 1
    print(counter, config)
    print("---------------------------------------------------------------------------------------------------")
    # createSequentialModel(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7],
    #                       "temp_model_{}".format(counter))
    acc = checkAccurancy("temp_model_{}".format(counter), X_test, y_test, 32)
    print("---------------------------------------------------------------------------------------------------")
    list_of_configs.append(config)
    list_of_acc.append(acc)

counter = len(list_of_configs)
starting = 109
while counter > 0:
    starting -= 1
    print(starting)
    counter -= 1
    print(list_of_configs[counter])
    print(list_of_acc[counter])
    print("---------------------------------------------------")