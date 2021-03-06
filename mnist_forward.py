# coding: utf-8
import tensorflow as tf
INPUT_NODE = 784  # 28x28个像素点
OUTPUT_NODE = 10  # 定义输出为10类，每个位置上显示为该类的概率
LAYER1_NODE = 500 # 隐藏层的节点个数


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1)) # 初始化w
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        # 将每个变量的正则化损失加入到总损失集合losses中
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    # 要对输出结果进行softmax处理使得它符合概率分布，所以不过relu函数

    return y

