import numpy as np


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''

    def __init__(self, shape_x):
        self.nr_of_neurons = shape_x[2]
        weights_shape = self.nr_of_neurons * self.nr_of_neurons + self.nr_of_neurons
        biases_shape = self.nr_of_neurons + 1
        # Weights
        self.weights = np.random.normal(size=weights_shape)
        # self.w1 = np.random.normal()
        # self.w2 = np.random.normal()
        # self.w3 = np.random.normal()
        # self.w4 = np.random.normal()
        # self.w5 = np.random.normal()
        # self.w6 = np.random.normal()

        # Biases
        self.biases = np.random.normal(size=biases_shape)
        # self.b1 = np.random.normal()
        # self.b2 = np.random.normal()
        # self.b3 = np.random.normal()

    def feedforward_for_testing(self, x):
        # x is a numpy array with 2 elements.
        # h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        # h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        # o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        list_of_h = []
        for neuron in range(0, self.nr_of_neurons):
            sum_val_weights = 0
            for weight_nr in range(0, self.nr_of_neurons):
                # TODO get electrode historu
                electrode_history = x[weight_nr]
                sum_val_weights += self.weights[neuron * self.nr_of_neurons + weight_nr] * electrode_history
            list_of_h.append(sigmoid(sum_val_weights + self.biases[neuron]))
        sum_o1 = 0
        for h in range(0, self.nr_of_neurons):
            sum_o1 += list_of_h[h] * self.weights[self.nr_of_neurons * self.nr_of_neurons + h]
        o1 = sigmoid(sum_o1 + self.biases[self.nr_of_neurons + 1])
        return o1

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        # h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        # h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        # o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        list_of_h = []
        list_of_h_sum = []
        for neuron in range(0, self.nr_of_neurons):
            sum_val_weights = 0
            for weight_nr in range(0, self.nr_of_neurons):
                electrode_history = x[weight_nr]
                sum_val_weights += self.weights[neuron * self.nr_of_neurons + weight_nr] * electrode_history
            list_of_h_sum.append(sum_val_weights + self.biases[neuron])
            list_of_h.append(sigmoid(sum_val_weights + self.biases[neuron]))
        sum_o1 = 0
        for h in range(0, self.nr_of_neurons):
            sum_o1 += list_of_h[h] * self.weights[self.nr_of_neurons * self.nr_of_neurons + h]
        o1 = sigmoid(sum_o1 + self.biases[self.nr_of_neurons])
        o1 = round((o1*4)-0.0000001)
        # d = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1],4: [0, 0, 0, 1]}
        # o1 = np.array(d[o1])
        return o1, sum_o1, list_of_h, list_of_h_sum

    def train(self, data, all_y_trues):
        '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                # sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                # h1 = sigmoid(sum_h1)
                #
                # sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                # h2 = sigmoid(sum_h2)
                #
                # sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                # o1 = sigmoid(sum_o1)
                # y_pred = o1
                y_pred, sum_o1, list_of_h, list_of_h_sum = self.feedforward(x)

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                # d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                # d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                # d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                #
                # d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                # d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_weights = np.copy(self.weights)
                d_biases = np.copy(self.biases)

                d_biases[self.nr_of_neurons] = deriv_sigmoid(sum_o1)
                list_d_ypred = []
                for neuron in range(0, self.nr_of_neurons):
                    current_h = self.nr_of_neurons * self.nr_of_neurons + neuron
                    d_weights[current_h] = list_of_h[neuron] * deriv_sigmoid(
                        sum_o1)
                    list_d_ypred.append(
                        self.weights[current_h] * deriv_sigmoid(sum_o1))

                # # Neuron h1
                # d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                # d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                # d_h1_d_b1 = deriv_sigmoid(sum_h1)
                #
                # # Neuron h2
                # d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                # d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                # d_h2_d_b2 = deriv_sigmoid(sum_h2)

                for neuron in range(0, self.nr_of_neurons):
                    for weight_nr in range(0, self.nr_of_neurons):
                        electrode_history = 0
                        d_weights[neuron * self.nr_of_neurons + weight_nr] = electrode_history * deriv_sigmoid(
                            list_of_h_sum[neuron])
                    d_biases[neuron] = deriv_sigmoid(list_of_h_sum[neuron])

                # --- Update weights and biases
                # Neuron h1
                # self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                # self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                # self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                #
                # # Neuron h2
                # self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                # self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                # self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                #
                # # Neuron o1
                # self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                # self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                # self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                for neuron in range(0, self.nr_of_neurons):
                    for weight_nr in range(0, self.nr_of_neurons):
                        current_wight = neuron * self.nr_of_neurons + weight_nr
                        self.weights[current_wight] -= learn_rate * d_L_d_ypred * list_d_ypred[neuron] * d_weights[
                            current_wight]
                    self.biases[neuron] -= learn_rate * d_L_d_ypred * list_d_ypred[neuron] * d_biases[neuron]

                for weight_nr in range(0, self.nr_of_neurons):
                    current_wight = self.nr_of_neurons * self.nr_of_neurons + weight_nr
                    self.weights[current_wight] -= learn_rate * d_L_d_ypred * d_weights[current_wight]
                self.biases[self.nr_of_neurons] -= learn_rate * d_L_d_ypred * d_biases[self.nr_of_neurons]

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                y_preds_transp = np.transpose(y_preds)
                loss = mse_loss(all_y_trues, y_preds_transp[0])
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset
# data = np.array([  # 32 2
#     [  # 100 3
#         [  # 22 6
#             [0.123], [0.143], [0.723], [0.923], [0.1023], [0.7723]
#         ],
#         [
#             [0.193], [0.149], [0.923], [0.929], [0.1923], [0.9723]
#         ],
#         [
#             [0.173], [0.743], [0.773], [0.723], [0.7023], [0.7727]
#         ]
#     ],
#     [  # 500
#         [  # 22
#             [0.523], [0.543], [0.723], [0.923], [0.5023], [0.7723]
#         ],
#         [
#             [0.593], [0.549], [0.923], [0.929], [0.5923], [0.9723]
#         ],
#         [
#             [0.573], [0.743], [0.773], [0.723], [0.7023], [0.7727]
#         ]
#     ]
# ])
#
# desired = data.reshape((2,3,6))
# desired = np.average(desired, axis=1)
# all_y_trues = np.array([
#     3,
#     0
# ])
#
# # Train our neural network!
# network = OurNeuralNetwork(desired.shape)
# network.train(desired, all_y_trues)
# emily = np.array([-7, -3])  # 128 pounds, 63 inches
# frank = np.array([20, 2])  # 155 pounds, 68 inches
# print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
# print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M
