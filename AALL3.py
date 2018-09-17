# #9/14
# import tensorflow as tf
# # import os
# #
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # 1
# # x_data = [3, 1, 40, 12]
# # y_data = [11, 2, 125, 38]
# # 2
# # x_data = [[1, 3, 2],
# #           [3, 4, 2],
# #           [5, 1, 2],
# #           [6, 2, 3],
# #           [2, 3, 5],
# #           [3, 7, 2]]
# # y_data = [[22],
# #           [32],
# #           [23],
# #           [35],
# #           [30],
# #           [48]]
# # 3
# # x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # y_data = [1, 1, 1, 1, 1, 0, 0, 0, 0]
# # 4
# # x_data = [[1, 2, 1, 1],
# #           [2, 1, 3, 2],
# #           [3, 1, 3, 4],
# #           [4, 1, 5, 5],
# #           [1, 7, 5, 5],
# #           [1, 2, 5, 6],
# #           [1, 6, 6, 6],
# #           [1, 7, 7, 7]]
# #
# # y_data = [[0, 0, 1],
# #           [0, 0, 1],
# #           [0, 0, 1],
# #           [0, 1, 0],
# #           [0, 1, 0],
# #           [0, 1, 0],
# #           [1, 0, 0],
# #           [1, 0, 0]]
#
# # X = tf.placeholder(tf.float32, shape=[None,4], name='X')
# # Y = tf.placeholder(tf.float32, shape=[None,3], name='Y')
# #
# # W = tf.Variable(tf.random_normal([4,3]))
# # b = tf.Variable(tf.random_normal([3]))
# #
# # Hy = tf.nn.softmax(tf.matmul(X,W)+b)
# # # cost = -Y * tf.log(Hy)
# # cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(Hy)))
# # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     for i in range(10000):
# #         _, cost_ = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
# #     print(sess.run(W), sess.run(b))
# #     predicted=tf.cast(Hy>0.5,dtype=tf.float32)
# #     print(sess.run(predicted,feed_dict={X:x_data}))
# #     print("\n\n")
# #     print(cost_)
#
#
#
#
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# x_data = [[0, 0],
#           [0, 1],
#           [1, 0],
#           [1, 1]]
# y_data = [[0],
#           [1],
#           [1],
#           [0]]
#
# X = tf.placeholder(tf.float32, shape=[None,2], name='X')
# Y = tf.placeholder(tf.float32, shape=[None,1], name='Y')
#
# W = tf.Variable(tf.random_normal([2,1]))
# b = tf.Variable(tf.random_normal([1]))
#
# Hy = tf.sigmoid(tf.matmul(X,W)+b)
# # cost = -Y * tf.log(Hy)
# cost = -(Y*tf.log(Hy)+-Y*tf.log(Hy))
# optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10000):
#         _, cost_ = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
#     print('W:')
#     print(sess.run(W))
#     print('\n\n')
#     print('b:')
#     print(sess.run(b))
#     print("\n\n")
#
#     predicted=tf.cast(Hy>0.5,dtype=tf.float32)
#     print("predicted:")
#     print(sess.run(predicted,feed_dict={X:x_data}))
#
#     print("\n\ncost:")
#     print(cost_)

# 9/17 mnist
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

Hy = tf.nn.softmax(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(Hy)))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(Hy, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

total_epoch = 15
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(total_epoch):
        sum_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

            sum_cost += cost_

        print('epoch:%4d,cost:%9f' % (epoch + 1, sum_cost))

    print("accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print("learning end")
