import tensorflow as tf
import cv2
import numpy as np
import os


def hfd(v, illuminate):
    pixel_intense = tf.reshape(v, shape=[-1])
    illuminate_map = tf.reshape(illuminate, shape=[-1])

    def _norm_illuminate_map(m):
        min_value = tf.math.reduce_min(m)
        max_value = tf.math.reduce_max(m)

        return (m - min_value) / (max_value - min_value)

    norm_illuminate_map = _norm_illuminate_map(illuminate_map)

    equal_tensor = tf.cast(tf.range(0, 256), tf.uint8)
    equal_tensor = tf.expand_dims(equal_tensor, 1)
    equal_tensor = tf.tile(equal_tensor, [1, tf.shape(pixel_intense)[0]])

    pixel_intense = tf.expand_dims(pixel_intense, 0)
    pixel_intense = tf.tile(pixel_intense, [256, 1])

    norm_illuminate_map = tf.expand_dims(norm_illuminate_map, 0)
    norm_illuminate_map = tf.tile(norm_illuminate_map, [256, 1])

    pixel_intense = tf.cast(pixel_intense, tf.uint8)

    logical_tensor = tf.equal(pixel_intense, equal_tensor)

    hfd_hist = tf.cast(logical_tensor, tf.float32) * norm_illuminate_map
    hfd_hist = tf.reduce_sum(hfd_hist, -1)
    sum_value = tf.reduce_sum(hfd_hist)

    hfd_hist = hfd_hist / sum_value
    return hfd_hist


def cdf(hfd_hist):
    temp = tf.ones(shape=[256, 256], dtype=tf.float32)
    mtx = tf.matrix_band_part(temp, 0, -1)
    hfd_hist = tf.tile(tf.expand_dims(hfd_hist, 0), [256, 1])

    cdf_hist = hfd_hist * mtx
    cdf_hist = tf.transpose(cdf_hist)
    cdf_hist = tf.reduce_sum(cdf_hist, 0)

    pad_cfd_hist = tf.pad(cdf_hist, [[tf.shape(cdf_hist)[0], 0]], 'SYMMETRIC')
    cdf_hist, _ = tf.split(pad_cfd_hist, [tf.shape(cdf_hist)[0], tf.shape(cdf_hist)[0]], 0)
    return cdf_hist


def map_pixel(image_tensor, cdf_hist):
    unique_input_intense, _ = tf.unique(tf.reshape(image_tensor, [-1]))
    unique_input_intense_sorted = tf.contrib.framework.sort(tf.cast(unique_input_intense, tf.int32))

    map_cdf_hist = tf.gather(cdf_hist, unique_input_intense_sorted)
    image_shape = tf.shape(image_tensor)
    map_level = map_cdf_hist * 255.0

    arr_map_level = map_level.numpy()

    arr_intense_sort = unique_input_intense_sorted.numpy()
    arr_map_level = arr_intense_sort * 1.3 + arr_map_level * 0.2

    for i in range(1, arr_map_level.shape[0]):
        if i < 150:
            if arr_map_level[i] - arr_map_level[i - 1] > 1.7 or arr_map_level[i - 1] > arr_map_level[i]:
                arr_map_level[i] = arr_map_level[i - 1] + 1.7
        else:
            if arr_map_level[i] - arr_map_level[i - 1] < 1.5 or arr_map_level[i - 1] > arr_map_level[i]:
                arr_map_level[i] = arr_map_level[i - 1] + 1.5

    arr_map_level_f = np.zeros(shape=[256], dtype=np.int32)
    arr_map_level = np.minimum(arr_map_level.astype(np.int32), 255)

    arr_map_level_f[unique_input_intense_sorted.numpy()] = arr_map_level

    image_tensor = tf.reshape(image_tensor, [-1])

    enhanced_im = tf.gather(arr_map_level_f, tf.cast(image_tensor, tf.int32))
    enhanced_im = tf.reshape(enhanced_im, image_shape)

    return enhanced_im


def fcn_enhancement(img_path):
    tf.enable_eager_execution()
    image = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)

    with tf.gfile.GFile('./fcn_frozen_graph.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

        resized_inputs = cv2.resize(image, (500, 375)).astype(np.float32)
        in_ = (2.0 / 255.0) * resized_inputs - 1.0
        input_tensor = graph.get_tensor_by_name('import/input:0')
        background = graph.get_tensor_by_name('import/Squeeze:0')

        with tf.Session(graph=graph) as sess:
            b_ = sess.run(background, feed_dict={input_tensor: [in_]})

    image_tensor = tf.cast(image, tf.float32) / 255.0
    hsv_tensor = tf.image.rgb_to_hsv(image_tensor)

    h, s, v = tf.split(hsv_tensor, [1, 1, 1], -1)

    v = tf.math.minimum(v, 1, 0)
    v = tf.cast(v * 255.0, tf.uint8)
    resize_v = tf.image.resize_images(v, (background.shape[0], background.shape[1]))

    hfd_hist = hfd(resize_v, b_)
    cdf_hist = cdf(hfd_hist)

    enhance_v = map_pixel(v, cdf_hist)
    enhance_v = tf.cast(enhance_v, tf.float32) / 255.0
    enhance_hsv = tf.concat([h, s, enhance_v], axis=-1)
    enhance_image = tf.image.hsv_to_rgb(enhance_hsv)
    enhance_image = tf.cast(tf.minimum(enhance_image, 1.0) * 255.0, tf.uint8)

    cur_img_path = os.path.split(img_path)[1]
    img_title, img_ext = os.path.splitext(cur_img_path)
    enhance_img_path = img_title + '_enhanced' + img_ext
    cv2.imencode(img_ext, enhance_image.numpy())[1].tofile(enhance_img_path)


if __name__ == '__main__':
    img_path = './kodim24.png'
    fcn_enhancement(img_path)
