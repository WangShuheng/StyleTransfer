import os
import numpy as np
import scipy.io
import scipy.misc
from scipy.misc import imread,imresize
import tensorflow as tf

OUTPUT_DIR = 'output/'
STYLE_IMAGE = 'images/scream.jpg'
CONTENT_IMAGE = 'images/test.jpg'

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3
NOISE_RATIO = 0.6
ITERATIONS = 1000

alpha = 1
beta = 500

VGG_Model = 'imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

def load_image(path):
    image = imread(path)
    image = imresize(image,(IMAGE_HEIGHT,IMAGE_WIDTH))
    image = np.reshape(image, ((1,) + image.shape))
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def build_net(net_type, net_in, net_weight_bias=None):
    if net_type == 'conv':
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net_in, net_weight_bias[0], strides=[1, 1, 1, 1], padding='SAME') , net_weight_bias[1]))
    elif net_type == 'pool':
        return tf.nn.avg_pool(net_in, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(path):
    net = {}
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype('float32'))
    net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])
    return net

def content_loss_func(contents, net):

    layers = CONTENT_LAYERS
    total_content_loss = 0.0
    for layer_name, weight in layers:
        p = contents[layer_name]
        x = net[layer_name]
        M = p.shape[1] * p.shape[2]
        N = p.shape[3]
        total_content_loss += (1. / (2 * N * M)) * tf.reduce_sum(tf.pow((x - p), 2))*weight

    total_content_loss /= float(len(layers))
    return total_content_loss


def gram_matrix(x, area, depth):

    x1 = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def style_loss_func(styles, net):

    layers = STYLE_LAYERS
    total_style_loss = 0.0
    i = 0
    for layer_name, weight in layers:
        a = styles[i][layer_name]
        x = net[layer_name]
        M = a.shape[1] * a.shape[2]
        N = a.shape[3]
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        total_style_loss += (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2)) * weight
        i+=1
    total_style_loss /= float(len(layers))
    return total_style_loss


def main():
    net = build_vgg19(VGG_Model)
    styles = [{} for i in STYLE_LAYERS]
    contents = {}
    best_loss = 1e20

    with tf.Session() as sess:
        content_img = load_image(CONTENT_IMAGE)
        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(content_img))
        for layer_name , weight in CONTENT_LAYERS:
            contents[layer_name] = sess.run(net[layer_name])
        sess.close()

    with tf.Session() as sess:
        style_img = load_image(STYLE_IMAGE)
        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(style_img))
        i = 0
        for layer_name , weight in STYLE_LAYERS:
            styles[i][layer_name] = sess.run(net[layer_name])
            i+=1
        sess.close()

    with tf.Session() as sess:
        cost_content = content_loss_func(contents, net)
        cost_style = style_loss_func(styles, net)
        total_loss = alpha * cost_content + beta * cost_style
        train_op = tf.train.AdamOptimizer(2.0).minimize(total_loss)

        content_image = load_image(CONTENT_IMAGE)
        noise_img = np.random.uniform(-20, 20,(1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
        noise_img = noise_img * NOISE_RATIO + content_image * (1 - NOISE_RATIO)

        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(noise_img))

        for it in range(ITERATIONS+1):
            sess.run(train_op)
            if it % 100 == 0:
                mixed_image = sess.run(net['input'])
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                filename = 'output/%d.png' % (it)
                save_image(filename, mixed_image)
        sess.close()

if __name__ == '__main__':
    main()
