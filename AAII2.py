# # import tensorflow as tf
# # import os
# #
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #
# # x_data = [1,2,3]
# # y_data = [2,4,6]
# #
# # X = tf.placeholder(tf.float32)
# # Y = tf.placeholder(tf.float32)
# # W = tf.Variable(tf.random_normal([1], -1, 1))
# # b = tf.Variable(tf.random_normal([1], -1, 1))
# #
# # hypothesis = W * X + b
# # cost = tf.reduce_mean(tf.square(hypothesis - Y))
# #
# # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #
# #     for i in range(1000):
# #         _, cost_, val,b_= sess.run([optimizer, cost, W,b], feed_dict={X: x_data, Y: y_data})
# #         if(i%100 is 0):
# #             print("try %d cost %f W=%f b=%f" % (i, cost_, val,b_))
#
#
# # import tensorflow as tf
# # import numpy as np
# #
# # import os
# #
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #
# # # x,y의 데이터 값
# # data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
# # x_data = [x_row[0] for x_row in data]
# # y_data = [y_row[1] for y_row in data]
# #
# # # a와 b의 값을 임의로 정함
# # a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
# # b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
# #
# # # y 시그모이드 함수의 방정식을 세움
# # y = 1 / (1 + np.e ** (a * x_data + b))
# #
# # # loss를 구하는 함수
# # loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))
# #
# # # 학습률 값
# # learning_rate = 0.5
# #
# # # loss를 최소로 하는 값 찾기
# # gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# #
# # # 학습
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #
# #     for i in range(60001):
# #         sess.run(gradient_decent)
# #         if i % 6000 == 0:
# #             print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, 바이어스 b = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))
# #
# # tf.sigmoid
#
# # import tensorflow as tf
# # import os
# #
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #
# # x_data=[[1,2],[2,4],[3,5],[4,1]]
# # y_data=[[7],[14],[19],[14]]
# #
# # X=tf.placeholder(tf.float32,shape=[None,2],name='X')
# # Y=tf.placeholder(tf.float32,shape=[None,1],name='Y')
# #
# # W=tf.Variable(tf.random_normal([2,1],-5,5))
# # b=tf.Variable(tf.random_normal([1],-5,5))
# #
# # Hypoth=tf.matmul(X,W)+b
# # cost=tf.reduce_mean(tf.square(Y-Hypoth))
# # optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# #
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #
# #     for i in range(30000):
# #         sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
# #
# #         if(i%100 is 0):
# #             print(i,sess.run(W),'\tb:',sess.run(b))
#
# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data=[1,2,3]
# y_data=[0,1,1]
#
# X=tf.placeholder(tf.float32,shape=[None],name='X')
# Y=tf.placeholder(tf.float32,shape=[None],name='Y')
#
# W=tf.Variable(tf.random_normal([1],-5,5))
# b=tf.Variable(tf.random_normal([1],-5,5))
#
# hypoth=tf.sigmoid(W*X+b)
# cost=-tf.reduce_mean(Y*tf.log(hypoth)+(1-Y)*tf.log(1-hypoth))
# optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#
# with tf.Session() as sess:
#
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(10000):
#         sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
#         if(i%100 is 0):
#             print(sess.run(W),sess.run(b))
#
#     predicted=tf.cast(hypoth>0.5,dtype=tf.float32)
#     print(sess.run(predicted,feed_dict={X:x_data}))
# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data = [
#     [1, 2, 1, 1],
#     [2, 1, 3, 2],
#     [3, 1, 3, 4],
#     [4, 1, 5, 5],
#     [1, 7, 5, 5],
#     [1, 2, 5, 6],
#     [1, 6, 6, 6],
#     [1, 7, 7, 7]]
#
# y_data = [
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [1, 0, 0]]
#
# X = tf.placeholder(tf.float32, shape=[None,4], name='X')
# Y = tf.placeholder(tf.float32, shape=[None,3], name='Y')
#
# W = tf.Variable(tf.random_normal([4,3]))
# b = tf.Variable(tf.random_normal([3]))
#
# H = tf.nn.softmax(tf.add(tf.matmul(X,W),b))
# cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(H)))
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         for i in range(30000):
#             sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
#
#             if(i%100 is 0):
#                 print("step",i,'\n W \n',sess.run(W),'\n b \n',sess.run(b),'\n\n')
#
#         predicted=tf.cast(H>0.5,dtype=tf.float32)
#         print(sess.run(predicted,feed_dict={X:x_data}))

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = [3, 1, 40, 12]
y_data = [11, 2, 125, 38]

X = tf.placeholder(tf.float32, shape=[None], name='X')
Y = tf.placeholder(tf.float32, shape=[None], name='Y')

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

Hy = W * X + b
cost = tf.reduce_mean(tf.square(Y - Hy))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if (i % 100 is 0):
            print(sess.run(W), sess.run(b))



