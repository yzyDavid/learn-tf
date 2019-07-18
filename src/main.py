import tensorflow as tf


class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.output = None

    def make_graph(self):
        with self.graph.as_default():
            self._make_model_part()

    def _make_model_part(self):
        with tf.name_scope("model"):
            input_vec = tf.compat.v1.placeholder(name='input_vec', dtype=tf.float32, shape=(1, 10))
            matrix_1 = tf.compat.v1.get_variable(name='matrix_1', dtype=tf.float32, shape=(10, 5),
                                                 initializer=tf.random_normal_initializer)
            self.output = tf.matmul(input_vec, matrix_1)


def main():
    model = Model()
    model.make_graph()
    writer = tf.compat.v1.summary.FileWriter(logdir='/home/yzy/log_dir', graph=model.graph)

    with model.graph.as_default():
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            value = sess.run(model.output, feed_dict={'model/input_vec:0': [[1, 2, 3, 4, 5, 5, 4, 3, 2, 1]]})
            writer.add_graph(model.graph)

    writer.flush()
    print(value)


if __name__ == '__main__':
    main()
