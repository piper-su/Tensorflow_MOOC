# coding: utf-8
# 神经网络优化（一）：损失函数的影响(3)
"""
交叉熵ce(CROSS ENTROPY): 表征两个概率分布之间的距离
H(y_,y)=-sum(y_*log(y))

ce=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-12, 1.0))
y小于1e-12为1e-12, 防止出现log(0)
y大于1时为1, 概率分布小于等于1

为了让前向传播计算出来的值满足概率分布，即让n分类的n个输出(y1,...,yn)都在[0,1]之间
且和为1，引入softmax()函数
softmax(yi)=exp(yi)/sum(exp(yj))
替换上面的交叉熵
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_1,1))
cem = tf.reduce_mean(ce)
"""
# 0. 导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1

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
loss_def = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_def)

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
            total_loss = sess.run(loss_def, feed_dict={x: X, y_: Y_})
            print("After {:d} training step(s), loss on all data is {:.5f}".format(i, total_loss))
            print("After training: w1=\n" + str(sess.run(w1)))
    # 输出训练后的参数
    print("After training: w1=\n"+str(sess.run(w1)))

"""
OUTPUT:
1. COST < PROFIT
[[1.0254971]
 [1.049255 ]]
 往大了预测
2. Otherwise
[[0.9613091]
 [0.9727003]]
"""