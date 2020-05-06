import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Example 1: session keeps track of the random state
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c))

# Example 2: each new session will start the random state all over again.
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c))

with tf.Session() as sess:
    print(sess.run(c))

# Example 3: with operation level random seed, each op keeps its own seed.
c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

# Example 4: graph level random seed
tf.set_random_seed(2)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))