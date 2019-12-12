import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time as time
from random import sample
import cv2

# 区分是train还是play
IS_TRAINING = True


def weight_variable(shape, std):
    initial = tf.truncated_normal(shape, stddev=std, mean=0)
    return tf.Variable(initial)


def bias_variable(shape, std):
    initial = tf.constant(std, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')


# 输入：100*100的灰度图片，前面的None是batch size，这里都为1
x = tf.placeholder(tf.float32, shape=[None, 50, 100, 1], name="x")
# 输出：一个浮点数，就是按压时间，单位s
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# 第一层卷积 12个feature map
W_conv1 = weight_variable([3, 3, 1, 32], 0.1)
b_conv1 = bias_variable([32], 0.1)
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积 24个feature map
W_conv2 = weight_variable([3, 3, 32, 64], 0.1)
b_conv2 = bias_variable([64], 0.1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([16192, 100], 0.1)
b_fc1 = bias_variable([100], 0.1)
h_pool2_flat = tf.reshape(h_pool2, [-1, 16192])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
learn_rate = tf.placeholder(tf.float32, name="learn_rate")

W_fc2 = weight_variable([100, 2], 0.1)
b_fc2 = bias_variable([2], 0.1)
y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 因输出直接是时间值，而不是分类概率，所以用平方损失
cross_entropy = tf.reduce_mean(tf.square(y_fc2-y_))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
tf_init = tf.global_variables_initializer()

saver_init = tf.train.Saver({"W_conv1": W_conv1, "b_conv1": b_conv1,
                             "W_conv2": W_conv2, "b_conv2": b_conv2,
                             "W_fc1": W_fc1, "b_fc1": b_fc1,
                             "W_fc2": W_fc2, "b_fc2": b_fc2})

def start_train(sess):
    '''
    开始训练
    '''
    train_arr = []
    file_dir = './data/train/'
    for root, dirs, files in os.walk(file_dir):
        for imgName in files:
            if '.png' in imgName:
                print('imgName', imgName)
                imgName = imgName.replace('.png', '')
                value = (float(imgName.split('_')[1]), float(
                    imgName.split('_')[2]))
                train_arr.append((imgName, value, readIm(imgName)))

    # 训练了多少次
    train_count = 0

    while True:
        train_count += 1
        randArr = sample(train_arr, 100)
        # print(randArr[0][0])
        x_in = []
        y_out = []
        for i in range(len(randArr)):
            (a, b, c) = randArr[i]
            x_in.append(c)
            y_out.append([round(float(b[0]), 7), round(float(b[1]), 7)])


        if len(x_in) > 0:
            # 每训练100个保存一次
            if train_count % 100 == 0:
                saver_init.save(sess, "./save5/mode.mod")
                # ————————————————这里只是打印出来看效果——————————————————
                x_in = [x_in[0]]
                y_out = [y_out[0]]
                y_result = sess.run(
                    y_fc2, feed_dict={x: x_in, keep_prob: 1})
                # loss 计算损失
                loss = sess.run(cross_entropy, feed_dict={
                                y_fc2: y_result, y_: y_out})
                print(str(train_count), "y_out:", y_out,
                      "y_result:", y_result, "loss:", loss)
            # —————————————————————————————————————————————————————
            # 使用x_in，y_out训练
            sess.run(train_step, feed_dict={
                x: x_in, y_: y_out, keep_prob: 0.6, learn_rate: 0.001})

            # —————————————————————————————————————————————————————


def start_play(sess, fileName):
    x_in = [readIm(fileName)]
    # 神经网络的输出
    y_result = sess.run(y_fc2, feed_dict={x: x_in, keep_prob: 1})
    y_result = int(y_result[0][0]*10)
    return y_result


def readIm(imgName):
    result = ''
    try:
        imgSrc = './data/train/'+imgName+'.png'
        img = cv2.imread(imgSrc)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = np.reshape(gray, (50, 100, 1))

        # im = Image.open(imgSrc)

        # # 转换为jpg
        # bg = Image.new("RGB", im.size, (255, 255, 255))
        # # bg.paste(im, im)
        # bg.save(r"./t.jpg")

        # img_data = tf.image.decode_jpeg(
        #     tf.gfile.FastGFile('./t.jpg', 'rb').read())
        # # 使用TensorFlow转为只有1通道的灰度图
        # img_data_gray = tf.image.rgb_to_grayscale(img_data)
        # x_in = np.asarray(img_data_gray.eval(), dtype='float32')

        # [0,255]转为[0,1]浮点
        # for i in range(len(x_in)):
        #     for j in range(len(x_in[i])):
        #         x_in[i][j][0] /= 255
        # result = x_in
    except:
        pass
    return result


def train(fileName):
    with tf.Session() as sess:
        sess.run(tf_init)
        # saver_init.restore(sess, "./save5/mode.mod")
        if IS_TRAINING:
            start_train(sess)
        else:
            return start_play(sess, fileName)


if __name__ == "__main__":
    if IS_TRAINING:
        train('')
    else:
        print('>>>', train('k1575791862588_173'))
