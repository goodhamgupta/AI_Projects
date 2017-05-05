from __future__ import absolute_import, division, print_function
import tflearn
import tensorflow as tf

def not_operation():
    # Logical NOT
    X = [[0.], [1.]]
    Y = [[1.], [0.]]

    # TF graph
    with tf.Graph().as_default():
        graph = tflearn.input_data(shape=[None,1])
        graph = tflearn.fully_connected(graph, 128, activation='linear')
        graph = tflearn.fully_connected(graph, 128, activation='linear')
        graph = tflearn.fully_connected(graph, 1, activation='sigmoid')
        graph = tflearn.regression(graph, optimizer='adam', learning_rate=0.1, loss='mean_square')

        # Model training
        model = tflearn.DNN(graph)
        model.fit(X, Y, n_epoch=5000, snapshot_epoch=False)
        prediction = model.predict([[1.]])
        print("Prediction: ", prediction)


def or_operation():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [[0.], [1.], [1.], [1.]]

    # Graph definition
    with tf.Graph().as_default():
        graph = tflearn.input_data(shape=[None, 2])
        graph = tflearn.fully_connected(graph, 128, activation='linear')
        graph = tflearn.fully_connected(graph, 1, activation='sigmoid')
        graph = tflearn.regression(graph, optimizer='adam', learning_rate=2.,
                               loss='mean_square')

        # Model training
        model = tflearn.DNN(graph)
        model.fit(X, Y, n_epoch=100, snapshot_epoch=False)
        prediction = model.predict([[0., 1.]])
        print("Prediction: ", prediction)

def xor_operation():
    # Function to simulate XOR operation using graph combo of NAND and OR
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y_nand = [[1.], [1.], [1.], [0.]]
    Y_or = [[0.], [1.], [1.], [1.]]

    with tf.Graph().as_default():
        graph = tflearn.input_data(shape=[None, 2])
        graph_nand = tflearn.fully_connected(graph, 32, activation='linear')
        graph_nand = tflearn.fully_connected(graph_nand, 32, activation='linear')
        graph_nand = tflearn.fully_connected(graph_nand, 1, activation='sigmoid')
        graph_nand = tflearn.regression(graph_nand, optimizer='sgd', learning_rate=2., loss='binary_crossentropy')

        graph_or = tflearn.fully_connected(graph, 32, activation='linear')
        graph_or = tflearn.fully_connected(graph_or, 32, activation='linear')
        graph_or = tflearn.fully_connected(graph_or, 1, activation='sigmoid')
        graph_or = tflearn.regression(graph_or, optimizer='sgd', learning_rate=2., loss='binary_crossentropy')

        graph_xor = tflearn.merge([graph_nand, graph_or], mode='elemwise_mul')

        # Model training
        model = tflearn.DNN(graph_xor)

        model.fit(X, [Y_nand, Y_or], n_epoch=100, snapshot_epoch=False)
        prediction = model.predict([[0., 1.]])
        print("Prediction: ", prediction)

if __name__ == '__main__':
    xor_operation()
