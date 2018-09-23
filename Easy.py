import tensorflow as tf
import os


def NoRedLine():  # 계속쓰기 힘들어서
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def g_v_init():  # 걍 보기에 길어서..?
    return tf.global_variables_initializer()


class InputLayer():
    def __init__(self, x_name, x_shape, y_name, y_shape):
        self.x_shape = x_shape
        self.x_name = x_name
        self.y_shape = y_shape
        self.y_name = y_name

        self.X, self.Y = Inputplace(self.x_name, self.x_shape), Inputplace(self.y_name, self.y_shape)


def Variable(name, shape):
    return tf.Variable(tf.random_normal(shape), name)


def Inputplace(name, shape):
    return tf.placeholder(tf.int32, shape, name)


class Layer():
    def __init__(self, W_name, W_tf_shape, b_name, b_tf_shape, X, Hyname):
        self.W_name = W_name
        self.W_tf_shape = W_tf_shape
        self.b_name = b_name
        self.b_tf_shape = b_tf_shape
        self.X = X
        self.Hyname = Hyname
        self.W = Insatane(self.W_name, self.W_tf_shape)
        self.b = Insatane(self.b_name, self.b_tf_shape)
        self.Hy = Hy(self.X, self.W, self.b, self.Hyname)


def Insatane(name, tf_shape):
    return Variable(tf_shape, name)


def Hy(X, W, b, Hyname):
    if (Hyname in "linear"):
        return tf.matmul(X, W) + b
    elif (Hyname in "simplesigmoid"):
        return tf.sigmoid(W * X +b)
    elif (Hyname in "matrixsigmoid"):
        return tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))


def cost(Y, Hy, Hyname):
    if (Hyname in "linear"):
        return tf.reduce_mean(tf.square(Y - Hy))
    elif (Hyname in "sigmoid"):
        return -tf.reduce_mean(Y * tf.log(Hy) + (1 - Y) * tf.log(1 - Hy))


def Optimizer(learning_rate, minimize_cost, Optimizer_name):
    if (Optimizer_name in "GradientDescentOptimizer"):
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(minimize_cost)





if __name__ is "Easy":
    print("Hello Easy!!!")
