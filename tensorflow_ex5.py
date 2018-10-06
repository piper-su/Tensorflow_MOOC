# coding: utf-8
# 神经网络优化（一）：损失函数的影响(1)
"""
MSE: 预测多或预测少的影响一样
"""
# 0. 导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
print(Y_)
# 1. 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)
print(y)
print(y_)
# 2. 定义损失函数和反向传播方法
# 定义损失函数为MSE， 反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 3. 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 30000
    # 打印未训练时的W
    print("Before training: w1=\n"+str(sess.run(w1)))

    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = BATCH_SIZE + start
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After {:d} training step(s), loss on all data is {:.5f}".format(i, total_loss))
            print("After training: w1=\n" + str(sess.run(w1)))
    # 输出训练后的参数
    print("After training: w1=\n"+str(sess.run(w1)))

"""
OUTPUT:
真实的关系是x1+x2+d=y
预测结果：
[[1.000973 ]
 [0.9977467]]
 都接近与1
"""
