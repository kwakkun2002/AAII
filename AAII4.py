# 9/23 12:28 레이어를 클래스로 구현
# 9/23 갑자기 안됨:오류!
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
Input = E.InputLayer('X', [None, 2], 'Y', [None, 1])

layer1 = E.Layer('W1', [2, 100], 'b1', [100], Input.X, "matrixsig")
layer2 = E.Layer('W2', [100, 100], 'b2', [100], layer1.Hy, "matrixsig")
layer3 = E.Layer('W3', [100, 100], 'b3', [100], layer2.Hy, "matrixsig")
layer4 = E.Layer('W4', [100, 1], 'b4', [1], layer3.Hy, "matrix")

cost = E.cost(Input.Y, layer3.Hy, "s")
optimizer = E.Optimizer(0.01, cost, "G")

with tf.Session() as sess:
    sess.run(E.g_v_init())
    for i in range(10000):
        sess.run(optimizer, feed_dict={Input.X: x_data, Input.Y: y_data})
        cost_ = sess.run(cost, feed_dict={Input.X: x_data, Input.Y: y_data})
    print(sess.run(layer4.Hy, feed_dict={Input.X: x_data}))