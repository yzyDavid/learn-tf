import tensorflow as tf


class Model(object):
    def __init__(self):
        pass

    def make_graph(self):
        with tf.name_scope("model"):
            input_vec = tf.placeholder(name='input_vec', dtype=tf.float32, shape=(1, 10))
            matrix_1 = tf.get_variable(name='matrix_1', dtype=tf.float32, shape=(10, 5),
                                       initializer=tf.random_normal_initializer)
            output = tf.matmul(input_vec, matrix_1)
        return output


def main():
    model = Model()
    g = model.make_graph()
    writer = tf.summary.FileWriter(logdir='/home/yzy/log_dir', graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        value = sess.run(g, feed_dict={'model/input_vec:0': [[1, 2, 3, 4, 5, 5, 4, 3, 2, 1]]})
        # writer.add_summary(value)
        writer.add_graph(tf.get_default_graph())
        writer.flush()
        print(value)


if __name__ == '__main__':
    main()
