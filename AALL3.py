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
# import tensorflow as tf  # 평화로운 텐서플로우 임포트
# from tensorflow.examples.tutorials.mnist import input_data  # mnist가져오기
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 원_핫을 키고?mnist에 데이터저장
#
# X = tf.placeholder(tf.float32, shape=[None, 784], name='X')  # X는 한줄에 784짜리 행렬 행의 개수 미정
# Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')  # Y는 0-9까지의 숫자이기 때문에 one_hot으로 10개
# # 원_핫이 약간 omr같은건가(요)?아직 softmax가 완벽하게이해...ㅠㅠ
#
# W = tf.Variable(tf.random_normal([784, 10]))  # 적절한 행렬곱으로 적절하게 구할수 있음
# b = tf.Variable(tf.random_normal([10]))  # 이것도 행렬로 알수있음
#
# Hy = tf.nn.softmax(tf.matmul(X, W) + b)  # 소프트맥스 가설
# cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(Hy)))  # 코스트
# optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)  # 코스트 최소화
#
# is_correct = tf.equal(tf.arg_max(Hy, 1), tf.arg_max(Y, 1))  # 근데 왜 1이 들어가죠?저 max가 최대값찾는
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  # 고동형이 cast는 map이랑 비슷하댔어!!아마 is_correct는 0이나 1이 오겠지
#
# total_epoch = 15  # 반복횟수
# batch_size = 100  # 자를 크기가 아닌 자를 횟수라고 해야하나
# total_batch = int(mnist.train.num_examples / batch_size)  # 이게 총 개수
# # 그러니까 데이터가 크니까 자른거임 근데 너무 자르면 학습이 이상하게 된다 그래서 적당히 자르는게 중요!
#
# with tf.Session() as sess:  # cpu가 신나는 세션열기
#     sess.run(tf.global_variables_initializer())  # 변수 초기화
#
#     for epoch in range(total_epoch):  # 반복하자 근데 굳이 15인 이유는??
#         sum_cost = 0  # 코스트를 계속 누적시키겠지...
#
#         for i in range(total_batch):  # 자른만큼 돌자
#             batch_x, batch_y = mnist.train.next_batch(batch_size)  # 값을 신기하게 받아옴;
#
#             _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})  # _는 안쓰고,cost_는 코스트
#
#             sum_cost += cost_  # 누적누적
#
#         print('epoch:%4d,cost:%9f' % (epoch + 1, sum_cost))  # 횟수와 코스트 출력
#
#     print("accuracy:",
#           sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))  # 얼마나 학습이 잘됬는지 보는 accuarcy
#     print("learning end")  # 이야 끝났다!!복습ㄱㄱ나머지도 주석ㄱㄱ
#     # 항상 느끼는건데 결국 cost랑 hy랑 accuracy만 바뀌는것 같은데...

# 9/19
# import tensorflow as tf
# # 역사 다층퍼셉트론의 등장l
# import numpy as np
#
# tf.set_random_seed(777)  # for reproducibility
# learning_rate = 0.1
#
# x_data = [[0, 0],
#           [0, 1],
#           [1, 0],
#           [1, 1]]
# y_data = [[0],
#           [1],
#           [1],
#           [0]]
# x_data = np.array(x_data, dtype=np.float32)
# y_data = np.array(y_data, dtype=np.float32)
#
# X = tf.placeholder(tf.float32, [None, 2])
# Y = tf.placeholder(tf.float32, [None, 1])
#
# W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
# b1 = tf.Variable(tf.random_normal([2]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
# b2 = tf.Variable(tf.random_normal([1]), name='bias2')
# hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
#
# # cost/loss function
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
#                        tf.log(1 - hypothesis))
#
# train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # Accuracy computation
# # True if hypothesis>0.5 else False
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# # Launch graph
# with tf.Session() as sess:
#     # Initialize TensorFlow variables
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         sess.run(train, feed_dict={X: x_data, Y: y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={
#                   X: x_data, Y: y_data}), sess.run([W1, W2]))
#
#     # Accuracy report
#     h, c, a = sess.run([hypothesis, predicted, accuracy],
#                        feed_dict={X: x_data, Y: y_data})
#     print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

# 9/21 다층 퍼셉트론 xor
# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]
# y_data = [
#     [0],
#     [1],
#     [1],
#     [0]
# ]
#
# X = tf.placeholder(tf.float32, shape=[None, 2])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W1 = tf.Variable(tf.random_normal([2, 100]), name='W1')
# b1 = tf.Variable(tf.random_normal([100]), name='b1')
# h1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1), b1))
#
# W2 = tf.Variable(tf.random_normal([100, 100]), name='W2')
# b2 = tf.Variable(tf.random_normal([100]), name='b2')
# h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1, W2), b2))
#
# W3 = tf.Variable(tf.random_normal([100, 100]), name='W3')
# b3 = tf.Variable(tf.random_normal([100]), name='b3')
# h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, W3), b3))
#
# W4 = tf.Variable(tf.random_normal([100, 1]), name='W4')
# b4 = tf.Variable(tf.random_normal([1]), name='b4')
# h4 = tf.nn.sigmoid(tf.add(tf.matmul(h3, W4), b4))
#
# cost = -tf.reduce_mean(Y * tf.log(h4) + (1 - Y) * tf.log(1 - h4))
# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(100000):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         cost_ = sess.run(cost, feed_dict={X: x_data, Y: y_data})
#
#         if (i % 100 is 0):
#             print(sess.run([W1, W2]))
#             print(cost_)
#             print()
#
#     print(sess.run(h4, feed_dict={X: x_data}))

# 9/22 숫자 맞추기 게임
# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data = [
#     [1, 2],
#     [3, 4],
#     [5, 6],
#     [7, 8],
#     [9, 10],
#     [10, 11]
# ]
# y_data = [
#     [2],
#     [12],
#     [30],
#     [56],
#     [90],
#     [110]
# ]
#
# X = tf.placeholder(tf.float32, shape=[None,2], name='X')
# Y = tf.placeholder(tf.float32, shape=[None,1], name='Y')
#
# W = tf.Variable(tf.random_normal([2,1]))
# b = tf.Variable(tf.random_normal([1]))
#
# Hy = tf.add(tf.matmul(X,W),b)
# cost = tf.reduce_mean(tf.square(Y - Hy))
# optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10000):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#     answer=sess.run(Hy,feed_dict={X:[[20,20]]})
#     print(int(answer))

# 9/22 실험 1.일반적인 로지스틱도 레이어를 늘리면 될까? 상식적으로,될것 같다.왜냐하면 상식적으로 xor이나 그냥이나 같기 때문이다
# 하지만 층이 여러개 필요한 상황에는 여러개를 쓰는것이다.
# 실험2.cost와 hy를 미리 만들순 없을까?함수로?ㅋㅋ됨,Easy.py만듬ㅋ
# import tensorflow as tf
# import Easy
#
# Easy.NoRedLine()
#
# x_data = [[1],
#           [2],
#           [3],
#           [4]]
#
# y_data = [[1],
#           [2],
#           [3],
#           [4]]
#
# X = Easy.Inputplace([None, 1], 'X')
# Y = Easy.Inputplace([None, 1], 'Y')
#
# W, b = Easy.Layer('W', [1, 1], 'b', [1])
# Hy = Easy.Hy(X, W, b, "l")
#
# cost = Easy.cost(Y, Hy, "l")
# optimizer = Easy.Optimizer(0.001, cost, "G")
#
# with tf.Session() as sess:
#     sess.run(Easy.g_v_init())
#
#     for i in range(30000):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         cost_ = sess.run(cost, feed_dict={X: x_data, Y: y_data})
#         if (i % 100 is 0):
#             print(i, sess.run(W), '\tb:', sess.run(b))
#     print(cost_)

# 한번 9/21일자 xor문제를 east로 번역해야겠다
# 9/21 다층 퍼셉트론 xor
import tensorflow as tf
import Easy as E

E.NoRedLine()

x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_data = [
    [0],
    [1],
    [1],
    [0]
]

# X = tf.placeholder(tf.float32, shape=[None, 2])
X = E.Inputplace([None, 2], 'X')
# Y = tf.placeholder(tf.float32, shape=[None, 1])
Y = E.Inputplace([None, 1], 'Y')

# # W1 = tf.Variable(tf.random_normal([2, 100]), name='W1')
# W1 = E.Store.Make_W([2, 100], 'W1')
# # b1 = tf.Variable(tf.random_normal([100]), name='b1')
# b1 = E.Store.Make_b([100], 'b1')
# # h1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1), b1))
W1,b1=E.Layer().Insatane('W1',[2,100],'b1',[100])
h1 = E.Layer().Hy(X, W1, b1,"s")

# # W2 = tf.Variable(tf.random_normal([100, 100]), name='W2')
# W2 = E.Store.Make_W([100, 100], 'W2')
# # b2 = tf.Variable(tf.random_normal([100]), name='b2')
# b2 = E.Store.Make_b([100], 'b2')
# # h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1, W2), b2))
# h2 = E.Hy.Make_Hy_sig(h1, W2, b2)
W2,b2=E.Layer().Insatane('W2',[100,100],'b2',[100])
h2=E.Layer().Hy(h1,W2,b2,"s")

# # W3 = tf.Variable(tf.random_normal([100, 100]), name='W3')
# W3 = E.Store.Make_W([100, 100], 'W3')
# # b3 = tf.Variable(tf.random_normal([100]), name='b3')
# b3 = E.Store.Make_b([100], 'b3')
# # h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, W3), b3))
# h3 = E.Hy.Make_Hy_sig(h2, W3, b3)
W3,b3=E.Layer().Insatane('W3',[100,100],'b3',[100])
h3=E.Layer().Hy(h2,W3,b3,"s")

# # W4 = tf.Variable(tf.random_normal([100, 1]), name='W4')
# W4 = E.Store.Make_W([100, 1], 'W4')
# # b4 = tf.Variable(tf.random_normal([1]), name='b4')
# b4 = E.Store.Make_b([1], 'b4')
# # h4 = tf.nn.sigmoid(tf.add(tf.matmul(h3, W4), b4))
# h4 = E.Hy.Make_Hy_sig(h3, W4, b4)
W4,b4=E.Layer().Insatane('W4',[100,1],'b4',[1])
h4=E.Layer().Hy(h3,W4,b4,"s")

cost = E.cost(Y,h4,"s")
optimizer=E.Optimizer(0.01,cost,"Gra")

with tf.Session() as sess:
    sess.run(E.g_v_init())
    for i in range(10000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        cost_ = sess.run(cost, feed_dict={X: x_data, Y: y_data})

        if (i % 100 is 0):
            print(sess.run([W1, W2]))
            print(cost_)
            print()

    print(sess.run(h4, feed_dict={X: x_data}))
