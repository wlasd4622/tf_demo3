import tensorflow as tf

x = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y_ = tf.placeholder(tf.float32)

y = W * x + b

lost = tf.reduce_mean(tf.square(y_-y))
optimizer = tf.train.GradientDescentOptimizer(0.0000001)

# train_step = tf.train.AdamOptimizer(0.5).minimize(lost)
train_step = optimizer.minimize(lost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

steps = 150000
for i in range(steps):
    xs = [i]
    ys = [100 * i]
    feed = {x: xs, y_: ys}
    sess.run(train_step, feed_dict=feed)
    if i % 500 == 0:
        print("After %d iteration:" % i, "W: %f" % sess.run(W), "b: %f" %
              sess.run(b), "lost: %f" % sess.run(lost, feed_dict=feed))
