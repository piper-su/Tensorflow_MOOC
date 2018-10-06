# coding: utf-8
# 神经网络优化（二）：学习率的影响
"""
学习率：每次参数更新的幅度
w(n+1)=w(n)-learning_rate*gradient
设损失函数loss=(w+1)^2，令初值为5，反响传播就是求最优w，即最小化loss对应的w值
学习率大了振荡不收敛，小了收敛速度慢

指数衰减学习率：根据BATCH_SIZE的轮数动态更新学习率
learning_rate = LEARNING_RATE_BASE*LEARNING_RATE_DECAY**(global_step/LEARNING_RATE_STEP)
                学习率基数、初值       学习率衰减率(0,1)  运行了几次BATCH_SIZE/多少轮更新一次学习率=总样本数/BATCH_SIZE
"""
import tensorflow as tf
LEARNING_RATE_BASE = 0.1  # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为总样本/BATCH_SIZE
# 运行了几轮BATCH_SIZE的计数器，初值为0，设为不可训练
global_step = tf.Variable(0, trainable=False)  # 不可训练变量
# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)
# straircase=True global_step/LRS 取整数，阶梯型衰减；否则平滑曲线

# 定义待优化参数W初值为5
W = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数loss
loss = tf.square(W+1)
# 定义反响传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                       global_step=global_step)
# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(W)
        loss_val = sess.run(loss)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        print("After {:d} steps: w is {:f}, loss is {:f}, "
              "learning rate is {:f}, global_step is {:f}.".format(i, w_val, loss_val,
                                                                   learning_rate_val, global_step_val))

"""
OUTPUT:
learning_rate=0.2
After 39 steps: w is -1.000000, loss is 0.000000.
learning_rate=2
After 39 steps: w is 72945990637077921792.000000, loss is inf. 震荡不收敛
learning_rate=0.02
After 39 steps: w is 0.172197, loss is 1.374046. 收敛速度慢
指数衰减学习率
After 39 steps: w is -0.995731, loss is 0.000018, learning rate is 0.066897, global_step is 40.000000.
"""