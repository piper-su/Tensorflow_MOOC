# coding: utf-8
# 保存模型
saver = tf.train.Sever() # 实例化saver对象
with tf.Session() as sess: # 在with结构for循环中一定轮数保存模型当前会话
    if i%轮数 == 0:
        saver.save(sess,
                   os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                   global_step=global_step)
# 加载模型
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(存储路径)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

# 实例化可还原滑动平均值
ema = tf.train.ExponentialMovingAverage(滑动平均基数)
ema_restore = ema.variable_to_restore()
saver = tf.train.Saver(ema_restore)

# 准确率计算方法
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
# argmax(y,1)返回y的每一行最大值的索引号返回
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))