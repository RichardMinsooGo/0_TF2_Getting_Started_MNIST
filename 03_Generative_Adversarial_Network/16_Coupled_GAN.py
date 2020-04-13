import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import scipy.ndimage.interpolation

from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/16_Coupled"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32
INPUT_SIZE = mnist.train.images.shape[1]
OUTPUT_SIZE = mnist.train.labels.shape[1]
NOISE_SIZE = 128
H_SIZE_01 = 256
eps = 1e-8
lr = 1e-3
d_steps = 3

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

X1 = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
X2 = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Gen    = tf.Variable(xavier_init([NOISE_SIZE, H_SIZE_01]))
W02_Gen_01 = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
W02_Gen_02 = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen    = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen_01 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))
B02_Gen_02 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variables
W01_Dis_01 = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W01_Dis_02 = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis    = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis_01 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B01_Dis_02 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis    = tf.Variable(tf.zeros(shape=[1]))
"""

W01_Gen    = tf.get_variable("W01_Gen", shape=[NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_01 = tf.get_variable("W02_Gen_01", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_02 = tf.get_variable("W02_Gen_02", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen    = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen_01 = tf.Variable(tf.random_normal([INPUT_SIZE]))
B02_Gen_02 = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis_01 = tf.get_variable("W01_Dis_01", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W01_Dis_02 = tf.get_variable("W01_Dis_02", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis    = tf.get_variable("W02_Dis", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis_01 = tf.Variable(tf.random_normal([H_SIZE_01]))
B01_Dis_02 = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis    = tf.Variable(tf.random_normal([1]))

# Build Generator Network.
def GENERATOR(z):
    _LAY01_Gen = tf.nn.relu(tf.matmul(z, W01_Gen) + B01_Gen)
    output_Gen1 = tf.nn.sigmoid(tf.matmul(_LAY01_Gen, W02_Gen_01) + B02_Gen_01)
    output_Gen2 = tf.nn.sigmoid(tf.matmul(_LAY01_Gen, W02_Gen_02) + B02_Gen_02)
    return output_Gen1, output_Gen2

def DISCRIMINATOR(X1, X2):
    _LAY01_Dis1 = tf.nn.relu(tf.matmul(X1, W01_Dis_01) + B01_Dis_01)
    _LAY01_Dis2 = tf.nn.relu(tf.matmul(X2, W01_Dis_02) + B01_Dis_02)
    output_Dis1 = tf.nn.sigmoid(tf.matmul(_LAY01_Dis1, W02_Dis) + B02_Dis)
    output_Dis2 = tf.nn.sigmoid(tf.matmul(_LAY01_Dis2, W02_Dis) + B02_Dis)
    return output_Dis1, output_Dis2

G_var_list = [W02_Gen_01, W02_Gen_02, B02_Gen_01, B02_Gen_02]
G_shared_var_list = [W01_Gen, B01_Gen]

D_var_list = [W01_Dis_01, W01_Dis_02, B01_Dis_01, B01_Dis_02]
D_shared_var_list = [W02_Dis, B02_Dis]

# Train D
G1_sample, G2_sample = GENERATOR(z)
D1_real, D2_real = DISCRIMINATOR(X1, X2)
D1_fake, D2_fake = DISCRIMINATOR(G1_sample, G2_sample)

D1_loss = -tf.reduce_mean(tf.log(D1_real + eps) + tf.log(1. - D1_fake + eps))
D2_loss = -tf.reduce_mean(tf.log(D2_real + eps) + tf.log(1. - D2_fake + eps))
D_loss = D1_loss + D2_loss

# Train G
G1_loss = -tf.reduce_mean(tf.log(D1_fake + eps))
G2_loss = -tf.reduce_mean(tf.log(D2_fake + eps))
G_loss = G1_loss + G2_loss

# D optimizer
D_opt = tf.train.AdamOptimizer(learning_rate=lr)
# Compute the gradients for a list of variables.
D_gv = D_opt.compute_gradients(D_loss, D_var_list)
D_shared_gv = D_opt.compute_gradients(D_loss, D_shared_var_list)
# Average by halfing the shared gradients
D_shared_gv = [(0.5 * x[0], x[1]) for x in D_shared_gv]
# Update
D_solver = tf.group(
    D_opt.apply_gradients(D_gv), D_opt.apply_gradients(D_shared_gv)
)

# G optimizer
G_opt = tf.train.AdamOptimizer(learning_rate=lr)
# Compute the gradients for a list of variables.
G_gv = G_opt.compute_gradients(G_loss, G_var_list)
G_shared_gv = G_opt.compute_gradients(G_loss, G_shared_var_list)
# Average by halfing the shared gradients
G_shared_gv = [(0.5 * x[0], x[1]) for x in G_shared_gv]
# Update
G_solver = tf.group(
    G_opt.apply_gradients(G_gv), G_opt.apply_gradients(G_shared_gv)
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

X_train = mnist.train.images
half = int(X_train.shape[0] / 2)

# Real image
X_train1 = X_train[:half]
# Rotated image
X_train2 = X_train[half:].reshape(-1, 28, 28)
X_train2 = scipy.ndimage.interpolation.rotate(X_train2, 90, axes=(1, 2))
X_train2 = X_train2.reshape(-1, 28*28)

# Cleanup
del X_train

def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]

def _GET_NOISE(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

i = 0

for episode in range(N_EPISODES):
    X1_mb, X2_mb = sample_X(X_train1, mb_size), sample_X(X_train2, mb_size)
    z_mb = _GET_NOISE(mb_size, NOISE_SIZE)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss],
        feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={z: z_mb}
    )

    if episode % 1000 == 0:
        sample1, sample2 = sess.run(
            [G1_sample, G2_sample], feed_dict={z: _GET_NOISE(8, NOISE_SIZE)}
        )

        samples = np.vstack([sample1, sample2])

        print("Episode : {:>5d}] [D_loss: {:2.5f}] [G_loss: {:2.5f}]"
              .format(episode, D_loss_curr, G_loss_curr))

        fig = plot(samples)
        plt.savefig(SAVE_DIR + '/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
