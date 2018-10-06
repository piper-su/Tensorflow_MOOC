# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data


def generateds():
    mnist = input_data.read_data_sets('./data', one_hot=True)
    # 返回各子集样本数
    print(mnist.train.num_examples)
    print(mnist.validation.num_examples)
    print(mnist.test.num_examples)
    # 返回标签和数据
    print(mnist.train.labels[0])
    print(mnist.train.images[0])
    # 取一小撮数据，准备喂入神经网络训练
    BATCH_SIZE = 20
    xs, ys = mnist.train.next_batch(BATCH_SIZE)
    print(xs.shape)
    print(ys.shape)

if __name__ == '__main__':
    generateds()