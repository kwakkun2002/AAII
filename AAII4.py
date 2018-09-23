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

X = E.Inputplace('X', [None, 2])
Y = E.Inputplace('Y', [None, 1])

# layer1 = E.Layer('W1', [2, 100], 'b1', [100], Input.X, "matrixsig")
W1 = E.Variable('W1', [2, 100])
b1 = E.Variable('b1', [100])
h1 = E.Hy(X, W1, b1, "matrix")

# layer2 = E.Layer('W2', [100, 100], 'b2', [100], layer1.Hy, "matrixsig")
W2 = E.Variable('W2', [100, 100])
b2 = E.Variable('b2', [100])
h2 = E.Hy(h1, W2, b2, "matrix")

# layer3 = E.Layer('W3', [100, 100], 'b3', [100], layer2.Hy, "matrixsig")
W3 = E.Variable('W3', [100, 100])
b3 = E.Variable('b3', [100])
h3 = E.Hy(h2, W3, b3, "matrix")

# layer4 = E.Layer('W4', [100, 1], 'b4', [1], layer3.Hy, "matrixsig")
W4 = E.Variable('W4', [100, 1])
b4 = E.Variable('b4', [1])
h4 = E.Hy(h3, W4, b4, "matrix")

cost = E.cost(Y, h4, "s")
optimizer = E.Optimizer(0.01, cost, "G")

with tf.Session() as sess:
    sess.run(E.g_v_init())
    for i in range(10000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        cost_ = sess.run(cost, feed_dict={X: x_data, Y: y_data})
    print(sess.run(h4, feed_dict={X: x_data}))
