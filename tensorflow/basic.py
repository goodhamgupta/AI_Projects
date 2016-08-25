# Basic operations in tensorflow
import tensorflow as tf

tf.add(1, 2)
# 3

tf.sub(2, 1)
# 1

tf.mul(2, 2)
# 4

tf.div(2, 2)
# 1

tf.mod(4, 5)
# 4

tf.pow(3, 2)
# 9

# x < y
tf.less(1, 2)
# True

# x <= y
tf.less_equal(1, 1)
# True

tf.greater(1, 2)
# False

tf.greater_equal(1, 2)
# False

tf.logical_and(True, False)
# False

tf.logical_or(True, False)
# True

tf.logical_xor(True, False)
# True


# 1D array is called tensor
# 2D tensors = nxn matrix

tensor_1 = tf.constant([[1., 2.], [3.,4]])

tensor_2 = tf.constant([[5.5,6.],[7.,8.]])

# create a matrix multiplication operation
output_tensor = tf.matmul(tensor_1, tensor_2)

# have to run the graph using a session
sess = tf.Session()

result = sess.run(output_tensor)
print(result)

sess.close()