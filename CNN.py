import numpy as np
import tensorflow as tf

def group_normalization(x, is_training, G = 16, ESP = 1e-5, scope = 'group_norm'):
    with tf.variable_scope(scope):
        # 1. [N, H, W, C] -> [N, C, H, W]
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.shape.as_list()

        # 2. reshape (group normalization)
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        
        # 3. get mean, variance
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        # 4. normalize
        x = (x - mean) / tf.sqrt(var + ESP)

        # 5. create gamma, bete
        gamma = tf.Variable(tf.constant(1.0, shape = [C]), dtype = tf.float32, name = 'gamma')
        beta = tf.Variable(tf.constant(0.0, shape = [C]), dtype = tf.float32, name = 'beta')

        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        # 6. gamma * x + beta
        x = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # 7. [N, C, H, W] -> [N, H, W, C]
        x = tf.transpose(x, [0, 2, 3, 1])
    return x

def normalize(x, is_training, norm_type, scope):
    if norm_type == 'group':
        x = group_normalization(x, is_training, scope = scope)
    else:
        x = tf.layers.batch_normalization(x, training = is_training, name = scope)
    return x

def Global_Average_Pooling(x, stride=1):
    return tf.layers.average_pooling2d(inputs=x, pool_size=np.shape(x)[1:3], strides=stride)

'''
Tensor("Relu:0", shape=(?, 112, 112, 32), dtype=float32)
Tensor("Relu_1:0", shape=(?, 112, 112, 32), dtype=float32)
Tensor("max_pooling2d/MaxPool:0", shape=(?, 56, 56, 32), dtype=float32)
Tensor("Relu_2:0", shape=(?, 56, 56, 64), dtype=float32)
Tensor("Relu_3:0", shape=(?, 56, 56, 64), dtype=float32)
Tensor("max_pooling2d_1/MaxPool:0", shape=(?, 28, 28, 64), dtype=float32)
Tensor("Relu_4:0", shape=(?, 28, 28, 128), dtype=float32)
Tensor("Relu_5:0", shape=(?, 28, 28, 128), dtype=float32)
Tensor("max_pooling2d_2/MaxPool:0", shape=(?, 14, 14, 128), dtype=float32)
Tensor("Relu_6:0", shape=(?, 14, 14, 256), dtype=float32)
Tensor("Relu_7:0", shape=(?, 14, 14, 256), dtype=float32)
Tensor("flatten/Reshape:0", shape=(?, 50176), dtype=float32)
'''
def Model(input_var, is_training, norm_type):
    x = input_var

    x = tf.layers.conv2d(x, 32, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_1')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.conv2d(x, 32, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_2')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    print(x)

    x = tf.layers.conv2d(x, 64, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_3')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.conv2d(x, 64, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_4')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    print(x)

    x = tf.layers.conv2d(x, 128, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_5')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.conv2d(x, 128, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_6')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    print(x)

    x = tf.layers.conv2d(x, 256, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_7')
    x = tf.nn.relu(x)
    print(x)

    x = tf.layers.conv2d(x, 256, [3, 3], 1, padding = 'SAME')
    x = normalize(x, is_training, norm_type, scope = 'norm_8')
    x = tf.nn.relu(x)
    print(x)
    
    # final layers
    x = Global_Average_Pooling(x)
    
    x = tf.layers.flatten(x)
    print(x)

    logits = tf.layers.dense(x, 5)
    predictions = tf.nn.softmax(logits)

    return logits, predictions

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 112, 112, 3])
    Model(input_var, False, 'group')
