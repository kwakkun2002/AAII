import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data =[[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
#
# x1=[xrow1[0] for xrow1 in data]
# x2=[xrow2[1] for xrow2 in data]
# y_data=[y_row[2] for y_row in data]
#
# a1=tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
# a2=tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
# b=tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))
#
# y=a1*x1+a2*x2+b
#
# rmse=tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))
#
# gd=tf.train.GradientDescentOptimizer(0.01).minimize(rmse)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for  step in range(2001):
#         sess.run(gd)
#         if step % 100==0:
#             print("epoch:%.f,rmse=%.04f,기울기 a1 =%.4f,기울기 a2= %.4f,b=%.4f" %(step,sess.run(rmse),sess.run(a1),sess.run(a2),sess.run(b)))

#-*- coding: utf-8 -*-

# x,y의 데이터 값
# data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
# x_data = [x_row[0] for x_row in data]
# y_data = [y_row[1] for y_row in data]
#
# # a와 b의 값을 임의로 정함
# a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
# b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
#
# # y 시그모이드 함수의 방정식을 세움
# y = 1/(1 + np.e**(a * x_data + b))
#
# # loss를 구하는 함수
# loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))
#
# # 학습률 값
# learning_rate=0.5
#
# # loss를 최소로 하는 값 찾기
# gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
# # 학습
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(60001):
#         sess.run(gradient_decent)
#         if i % 6000 == 0:
#             print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, 바이어스 b = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))

#-*- coding: utf-8 -*-


# 실행할 때마다 같은 결과를 출력하기 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x,y의 데이터 값
x_data = np.array([[2, 3],[4, 3],[6, 4],[8, 6],[10, 7],[12, 8],[14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1,1]).reshape(7, 1)

# 입력 값을 플래이스 홀더에 저장
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 a와 bias b의 값을 임의로 정함.
a = tf.Variable(tf.random_uniform([2,1], dtype=tf.float64)) # [2,1] 의미: 들어오는 값은 2개, 나가는 값은 1개
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# y 시그모이드 함수의 방정식을 세움
y = tf.sigmoid(tf.matmul(X, a) + b)

# 오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# 학습률 값
learning_rate=0.1

# 오차를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))


# 어떻게 활용하는가
    new_x = np.array([7, 6.]).reshape(1, 2)  #[7, 6]은 각각 공부 시간과 과외 수업수.
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부 시간: %d, 개인 과외 수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성: %6.2f %%" % (new_y*100))







