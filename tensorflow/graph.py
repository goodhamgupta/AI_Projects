# Testing graphs in tensorflow

import tensorflow as tf

new_graph = tf.Graph()
with new_graph.as_default():
    new_g_const = tf.constant([1.,2.])
    assert new_g_const.graph is new_graph

default_g = tf.get_default_graph()

normal = tf.truncated_uniform([4, 4, 4], mean=0.0, stddev=1.0)
session = tf.Session()
result = session.run(normal)
print result
session.close()
