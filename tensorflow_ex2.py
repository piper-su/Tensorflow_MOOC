# coding:utf-8
# 两层简单神经网络（全连接）
"""
生成计算图，
变量初始化（待优化的变量定义为tf.Variable）,
前向传播，
会话计算结果
"""
import tensorflow as tf

# 定义输入和参数
x = tf.constant([[0.7, 0.5]]) # 一行两列的二维张量
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 进行变量初始化
    sess.run(init_op)
    print("y is "+str(sess.run(y)))  # 计算图节点运算

"""
OUTPUT:
[[3.0904665]]
"""