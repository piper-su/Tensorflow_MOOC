# coding: utf-8
# tensorflow_ex10的模块化实现————前向传播
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf_ex10_generate
import tf_ex10_forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01
MOVING_AVERAGE_DECAY = 0.999

def backward(sample_num, feature_num):
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = tf_ex10_generate.generateds(sample_num, feature_num)

    y = tf_ex10_forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        sample_num/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        loss_total,
        global_step=global_step
    )

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step
    )
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % sample_num
            end = start + BATCH_SIZE
            sess.run(
                train_op,
                feed_dict={x: X[start:end], y_: Y_[start:end]}
            )
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print("After {:d} steps, loss is {:f}".format(i, loss_v))

        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__ == '__main__':
    sample_num = 300
    feature_num = 2
    backward(sample_num, feature_num)
