import numpy as np

training_set_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,0]])
training_set_outputs = np.array([[0, 1, 1, 0]]).T

class NeuralNetwork():
    def __init__(self):
        # Seed random number generator to return the same random number every time the program is executed.
        np.random.seed(1)

        # Single neuron. 3 inputs and 1 output. Generate 3x1 matrix with values between -1 and 1 and mean 0
        self.synaptic_weights = (2 * np.random.random((3,1))) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x*(1-x)


    def think(self, inputs):
        return self._sigmoid(np.dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for i in range(iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = np.dot(training_set_inputs.T, error * self._sigmoid_derivative(output))

            self.synaptic_weights += adjustment

            if (i%1000) == 0:
                print ("Error after {} iterations is: {}".format(str(i), np.mean(np.abs(error))))


x = NeuralNetwork()
x.train(training_set_inputs, training_set_outputs, 10000)
test = np.array([[1, 0, 1]])
print ("Value predicted for {} is: {}".format(test, str(x.think(test))))
