# coding: utf-8
# 神经网络优化（三）：滑动平均（影子值）
"""
滑动平均：记录了每个参数一段时间过往值的平均，增加了泛化性
针对所有参数：w和b(像给参数加了影子，参数变化，影子缓慢跟随)
影子=衰减率*影子+（1-衰减率）*参数  影子初值=参数初值
衰减率=min{MOVING_AVERAGE_DECAY, (1+轮数)/(10+轮数)}

ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op=ema.apply(tf.trainable_variables()) 运行此句，所有待优化的参数求滑动平均
tf.trainable_variables()将所有待训练的参数汇总成列表

将训练过程和滑动平均绑定成一个训练节点
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train')

ema.average(参数) 返回某些参数的滑动平均
"""

import tensorflow as tf
# 1. 定义变量及滑动平均类
# 定义一个32位浮点变量，初值为0.0，这个代码不断更新w1的参数，优化w1参数，滑动平均做了个w1的影子
w1 = tf.Variable(0, dtype=tf.float32)
# 定义num_updates（NN迭代轮数），初始值为0，不可被优化，这个参数不训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，给衰减率为0.99，当前轮数为global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# ema.apply后的括号里是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求
# 滑动平均值
# 在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
# ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

# 2. 查看不同迭代中变量的取值变化
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 用ema.average(w1)获得w1的滑动平局值，要运行多个节点，作为列表中的元素列出，写在sess.run中
    # 打印当前参数w1和w1滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 每次sess.run会更新一次w1的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))



"""
[0.0, 0.0]
[10.0, 0.81818163]
[10.0, 1.569421]
[10.0, 2.2591956]
[10.0, 2.892534]
[10.0, 3.4740539]
[10.0, 4.0079947]
[10.0, 4.4982495]
[10.0, 4.948393]
[10.0, 5.3617063]
[10.0, 5.741203]
[10.0, 6.0896497]
[10.0, 6.4095874]
[10.0, 6.703348]
[10.0, 6.973074]
[10.0, 7.2207313]
[10.0, 7.448126]
[10.0, 7.6569157]
[10.0, 7.8486223]
[10.0, 8.024644]

影子在逼近w1
"""