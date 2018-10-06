# coding: utf-8
"""
对单层神经网络进行初步搭建
对运算节点进行运算用session
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x, w)
print(y)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 进行变量初始化
    sess.run(init_op)
    a = sess.run(y)
    print(a)

"""
OUTPUT
Tensor("MatMul:0", shape=(1, 1), dtype=float32)
[[11.]]
"""