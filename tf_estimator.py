import tensorflow as tf

def input_fn(filanames, batch_size=256, num_epochs=None, perform_shuffle=False):
    """
    通过datasets读取数据
    :param filanames: 文件名，例如tfrecord文件名
    :param batch_size: batch_size 大小
    :param num_epochs: epoch次数
    :param perform_shuffle: 是否乱序
    :return: tensor格式的，一个batch的数据。
    """
    def _parse_fn(record):
        features = {
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, features)
        # image
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [28, 28])
        # label
        label = tf.cast(parsed["label"], tf.int64)
        return {"image": image}, label
    dataset = tf.data.TFRecordDataset(filanames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    # ======== 解析参数部分 ======== #
    learning_rate = params["learning_rate"]

    # ======== 网络结构部分 ======== #
    # input
    X = tf.cast(features["image"], tf.float32, name="input_net")
    X = tf.reshape(X, [-1, 28*28]) / 255
    # DNN
    deep_inputs = X
    deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=128)
    deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=64)
    y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=10)
    # output
    y = tf.reshape(y_deep, [-1, 10])
    pred = tf.nn.softmax(y, name="soft_max")

    # ======== 如果是 predict 任务 ======== #
    predictions = {"prob": pred}

