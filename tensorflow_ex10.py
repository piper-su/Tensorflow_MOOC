# coding: utf-8
# 神经网络优化（四）：正则化缓解过拟合
"""
正则化在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声（一般不正则化b）
loss = loss(y,y_)            + REGULARIZER*                            loss(w)
       模型中所有参数的损失函数  用超参数给出w在总loss中的比例，则正则化权重  需要正则化的参数
loss(w)=tf.contrib.layers.l1_regularizer(REGULARIZER)(w)  l1正则化
loss(w)=tf.contrib.layers.l2_regularizer(REGULARIZER)(w)  l2正则化

tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(w))
把内容加到集合对应位置做加法
loss = cem+tf.add_n(tf.get_collection('losses'))
"""
# 0. 导入模块，生成模拟数据集
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
BATCH_SIZE = 30
seed = 2
# 基于seed产生随机数
rdm = np.random.RandomState(seed)
# 随机数返回300行2列的矩阵，表示300组坐标点(x0,x1)作为输入数据集
X = rdm.randn(300, 2)
# 从X这300X2的矩阵中取出一行，判断如果两个坐标平方和小于2，给Y赋值为1，否则赋值为0
# 作为输入数据集的标签（正确答案）
Y_ = [int(x0**2+x1**2<2) for (x0, x1) in X]
# 遍历Y中的每个元素，1赋值'red'，0赋值'blue'，这样可视化显示时可以直观区分
Y_c = [['red' if y else 'blue'] for y in Y_]
# 对数据集X和标签Y进行shape整理，第一个元素为-1表示，随第二个参数计算得到，第二个元素表示多少列，
# 把X整列成n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
# print(X)
# print(Y_)
# print(Y_c)
# 用plt.scatter画出数据X各行中第0列和第1列元素的点，用各行Y_c对应的值表示颜色
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


# 定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses',
                         tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2)+b2  # 输出层不过激活函数

# 定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义反向传播方法：不包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After {:d} steps, loss is {:f}".format(i, loss_mse_v))
    # xx 在-3到3之间步长为0.01, yy在-3到3之间步长为0.01生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx,yy拉直，合并成一个2列的矩阵，得到一个网格坐标点集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点喂入神经网络， probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    # probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n" + str(sess.run(w1)))
    print("b1:\n" + str(sess.run(b1)))
    print("w2:\n" + str(sess.run(w2)))
    print("b2:\n" + str(sess.run(b2)))

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 定义反向传播方法：包含正则化
rain_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print("After {:d} steps, loss is {:f}".format(i, loss_v))
    # xx 在-3到3之间步长为0.01, yy在-3到3之间步长为0.01生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx,yy拉直，合并成一个2列的矩阵，得到一个网格坐标点集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点喂入神经网络， probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    # probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n" + str(sess.run(w1)))
    print("b1:\n" + str(sess.run(b1)))
    print("w2:\n" + str(sess.run(w2)))
    print("b2:\n" + str(sess.run(b2)))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()