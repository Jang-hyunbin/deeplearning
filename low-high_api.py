###################low api##################
W1 = tf.Variable(tf.random_normal([3, 3, 4, 64], stddev=0.01), name = 'pf_w1')
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv1')
L1 = tf.nn.relu(L1, name = 'pf_relu1')

###################high api#################

def conv(x_input, num_filter, kernel_size, name_conv1, name_relu1):
    
    conv = tf.layers.conv2d(inputs = x_input, filters = num_filter, kernel_size=[kernel_size,kernel_size], padding="SAME", strides=1, name = name_conv1)
    relu = tf.nn.relu(conv, name = name_relu1)

    return relu

conv1 = conv(X_img, 64, 3, 'pf_conv1', 'pf_relu1')






