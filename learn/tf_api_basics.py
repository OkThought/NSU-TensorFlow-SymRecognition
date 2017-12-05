# silence the warning about unused CPU instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32, name='weight')
b = tf.Variable([-.3], dtype=tf.float32, name='bias')
x = tf.placeholder(dtype=tf.float32, name='x')
y = tf.placeholder(dtype=tf.float32, name='y')
linear_model = W*x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))
