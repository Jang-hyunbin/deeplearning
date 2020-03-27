import tensorflow as tf
import numpy as np
import random
import time

np.random.seed(5)

a = 49000
batch_size = 4900
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
#keep_prob = tf.placeholder(tf.float32)

train = np.loadtxt('train_PF.csv', delimiter = ',', dtype = np.float32)
test = np.loadtxt('test_PF.csv', delimiter = ',', dtype = np.float32)
P = np.loadtxt('merge_print_PF.csv', delimiter = ',', dtype = np.float32)

x_data = train[1:49001,1:-1]
x_data = x_data.reshape([-1,8,8,4])

max_x_0 = np.max(x_data[:,:,:,0]); min_x_0 = np.min(x_data[:,:,:,0])
max_x_1 = np.max(x_data[:,:,:,1]); min_x_1 = np.min(x_data[:,:,:,1])
max_x_2 = np.max(x_data[:,:,:,2]); min_x_2 = np.min(x_data[:,:,:,2])
max_x_3 = np.max(x_data[:,:,:,3]); min_x_3 = np.min(x_data[:,:,:,3])

norm0=(x_data[:,:,:,0] - min_x_0)/(max_x_0 - min_x_0)
norm1=(x_data[:,:,:,1] - min_x_1)/(max_x_1 - min_x_1)
norm2=(x_data[:,:,:,2] - min_x_2)/(max_x_2 - min_x_2)
norm3=(x_data[:,:,:,3] - min_x_3)/(max_x_3 - min_x_3)

x_data[:,:,:,0]=norm0
x_data[:,:,:,1]=norm1
x_data[:,:,:,2]=norm2
x_data[:,:,:,3]=norm3

x_data = x_data.reshape([-1,256])

x_test = test[1:7001,1:-1]
x_test = x_test.reshape([-1,8,8,4])

norm0=(x_test[:,:,:,0] - min_x_0)/(max_x_0 - min_x_0)
norm1=(x_test[:,:,:,1] - min_x_1)/(max_x_1 - min_x_1)
norm2=(x_test[:,:,:,2] - min_x_2)/(max_x_2 - min_x_2)
norm3=(x_test[:,:,:,3] - min_x_3)/(max_x_3 - min_x_3)

x_test[:,:,:,0]=norm0
x_test[:,:,:,1]=norm1
x_test[:,:,:,2]=norm2
x_test[:,:,:,3]=norm3
x_test=x_test.reshape([-1,256])

x_P = P[1:14001,1:-1]
x_P = x_P.reshape([-1,8,8,4])

norm0=(x_P[:,:,:,0] - min_x_0)/(max_x_0 - min_x_0)
norm1=(x_P[:,:,:,1] - min_x_1)/(max_x_1 - min_x_1)
norm2=(x_P[:,:,:,2] - min_x_2)/(max_x_2 - min_x_2)
norm3=(x_P[:,:,:,3] - min_x_3)/(max_x_3 - min_x_3)

x_P[:,:,:,0]=norm0
x_P[:,:,:,1]=norm1
x_P[:,:,:,2]=norm2
x_P[:,:,:,3]=norm3
x_P=x_P.reshape([-1,256])

y_data = train[1:49001,[-1]]
y_data = y_data.reshape([-1,1])
max_y = np.max(y_data)
min_y = np.min(y_data)
y_data = (y_data - min_y)/(max_y - min_y)

y_test = test[1:7001,[-1]]

y_test = y_test.reshape([-1,1])
y_test = (y_test - min_y)/(max_y - min_y)

y_P = P[1:14001,[-1]]
y_P = y_P.reshape([-1,1])
y_P = (y_P - min_y)/(max_y - min_y)

# input place holders
X = tf.placeholder(tf.float32, [None, 256], name = 'pf_X')
X_img = tf.reshape(X, [-1, 8, 8, 4], name = 'pf_X_img')
Y = tf.placeholder(tf.float32, [None, 1], name = 'pf_Y')
#phase = tf.placeholder(tf.bool, name='pf_phase')

def conv(x_input, num_filter, kernel_size, name_conv1, name_relu1):
    
    #conv = tf.layers.conv2d(inputs = x_input, filters = num_filter, kernel_size=[kernel_size,kernel_size], padding="SAME", strides=1, name = name_conv1,
    #                        kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=regularizer)
    conv = tf.layers.conv2d(inputs = x_input, filters = num_filter, kernel_size=[kernel_size,kernel_size], padding="SAME", strides=1, name = name_conv1)
    relu = tf.nn.relu(conv, name = name_relu1)

    return relu

def conv_res(x_input, shortcut, num_filter, kernel_size, name_conv1, name_relu1):
    
    #conv = tf.layers.conv2d(inputs = x_input, filters = num_filter, kernel_size=[kernel_size,kernel_size], padding="SAME", strides=1, name = name_conv1,
    #                        kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=regularizer)
    conv = tf.layers.conv2d(inputs = x_input, filters = num_filter, kernel_size=[kernel_size,kernel_size], padding="SAME", strides=1, name = name_conv1)
    relu = tf.nn.relu(conv + shortcut, name = name_relu1)

    return relu

def pool(x_input, name_pool):
    pool = tf.layers.max_pooling2d(inputs= x_input, pool_size=[2,2], padding="SAME",strides=2,name=name_pool)
    
    return pool

lambda_reg = 0.0
regularizer = tf.contrib.layers.l2_regularizer(lambda_reg)

conv1 = conv(X_img, 64, 3, 'pf_conv1', 'pf_relu1')
conv2 = conv(conv1, 64, 3, 'pf_conv2', 'pf_relu2')
conv3 = conv(conv2, 64, 3, 'pf_conv3', 'pf_relu3')
conv4 = conv_res(conv3, conv1, 64, 3, 'pf_conv4', 'pf_relu4')

pool1 = pool(conv4, 'pf_pool1')

W4_up = tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=0.01), name = 'pf_w4_up')
conv4_up = tf.nn.conv2d(pool1, W4_up, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv4_up')

#conv4_up = conv(pool1, 128, 1, 'pf_conv4_up', 'pf_relu4_up')

conv5 = conv(conv4_up, 128, 3, 'pf_conv5', 'pf_relu5')
conv6 = conv_res(conv5, conv4_up, 128, 3, 'pf_conv6', 'pf_relu6')
conv7 = conv(conv6, 128, 3, 'pf_conv7', 'pf_relu7')
conv8 = conv(conv7, 128, 3, 'pf_conv8', 'pf_relu8')
conv9 = conv_res(conv8, conv6, 128, 3, 'pf_conv9', 'pf_relu9')

pool2 = pool(conv9, 'pf_pool2')

flat = tf.reshape(pool2, [-1,2*2*128], name = 'pf_flat')

dense1 = tf.layers.dense(inputs=flat, units=256, name = 'pf_dense1',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())#, kernel_regularizer=regularizer)
dense2 = tf.layers.dense(inputs=dense1, units=128, name = 'pf_dense2',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())#, kernel_regularizer=regularizer)
dense3 = tf.layers.dense(inputs=dense2, units=64, name = 'pf_dense3',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())#, kernel_regularizer=regularizer)
hypothesis = tf.layers.dense(inputs=dense3, units=1, name = 'pf_hypothesis',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())#, kernel_regularizer=regularizer)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# define cost/loss & optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
#train = optimizer.minimize(cost)
r=random.randint(0,154)
diff = (hypothesis - Y)*100000
rms = tf.sqrt(tf.reduce_mean(tf.square(diff)))
diff_max = tf.reduce_max(diff)
diff_min = tf.reduce_min(diff)

per = (hypothesis - Y)/(Y + min_y/(max_y - min_y))*100
mae = tf.reduce_mean(tf.abs(per))
rms_per = tf.sqrt(tf.reduce_mean(tf.square(per)))
per_max = tf.reduce_max(per)
per_min = tf.reduce_min(per)

# Launch the graph in a session.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
# Initializes global variables in the graph.
start_time = time.time()
sess.run(tf.global_variables_initializer())
# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver(max_to_keep=30)

out = open('M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2.out','w')
#out=open('ML_9layer.out','a')
#out.write('#-----------------------------------------------------------------------------------\n')
#out.write('#-----------------------------------------------------------------------------------\n')
#out.write('#-----------------------------------------------------------------------------------\n')
#out.close ()

a=np.logspace(1, 4, 100, endpoint=True, dtype=int)
b=np.logspace(4, 6, 200, endpoint=True, dtype=int)
c=np.logspace(6, 6.30103, 41, endpoint=True, dtype=int)

stop_count = 0
stop_cost = 20

for epoch in range(10001):
    step = epoch
    
    p = np.random.permutation(len(x_data))
    x_data=x_data[p]
    y_data=y_data[p]
    total_batch = int(len(x_data) / batch_size)
    avg_cost_test = avg_cost_train = 0

    for i in range(total_batch):
        #cost_val, hy_val, _ = sess.run(
        #[cost, hypothesis, optimizer], feed_dict = {X: x_data[i*batch_size:(i+1)*batch_size], Y: y_data[i*batch_size:(i+1)*batch_size]})
        opt = sess.run(optimizer, feed_dict = {X: x_data[i*batch_size:(i+1)*batch_size], Y: y_data[i*batch_size:(i+1)*batch_size]})
    
    if epoch % 10 == 0:
        cost_train, hy_val, rms_per_train, mae_train = sess.run(
        [cost, hypothesis,rms_per,mae], feed_dict={X: x_data, Y: y_data})#, phase : 1})

        cost_test, rms_test, rms_per_test, per_max_test, per_min_test, mae_test = sess.run(
        [cost, rms, rms_per, per_max, per_min, mae], feed_dict={X: x_test, Y: y_test})#, phase : 0})

        #out = open('M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test1.out','a')
        print(epoch,'epoch' , ' cost_train', cost_train, "cost_test ", cost_test, per_max_test, per_min_test, rms_per_test, rms_per_train, mae_test, mae_train)
        out.write('%d Cost_train %e  Cost_test %e %e %e %e %e %e %e\n' %(epoch,cost_train,cost_test,per_max_test,per_min_test,rms_per_test, rms_per_train, mae_test, mae_train))
        #out.close ()
        saver.save(sess, 'model/M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2/M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2', global_step=epoch, write_meta_graph=False)
        
        if stop_cost >= cost_test:
            stop_cost = cost_test
            stop_count = 0
        else:
            stop_count += 1

        if step > 1000 & stop_count == 10:
            break
        
out.write('time elapsed:  {:.2f}s \n'.format(time.time() - start_time))

out.close ()
print('Trained Model Saved.')
print("train: cost    per_max     per_min     rms_per    \n    ", sess.run([cost, per_max, per_min, rms_per, mae], feed_dict={X: x_data, Y: y_data}))#, phase : 1}))

#print("train: cost    rms_max     rms_min     rms    \n    ", sess.run([cost, diff_max, diff_min, rms], feed_dict={X: x_data, Y: y_data}))
print("test : cost    per_max     per_min     rms_per    \n    ", sess.run([cost, per_max, per_min, rms_per, mae], feed_dict={X: x_test, Y: y_test}))#, phase : 0}))
print("time elapsed: {:.2f}s".format(time.time() - start_time))
print_hypo = hypothesis * (max_y - min_y) + min_y
print_Y = Y * (max_y - min_y) + min_y

saver.restore(sess, 'model/M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2/M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2-%d' %(step))
out = open('M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2_diff_%d.out' %(step),'w')

cost_test, hypo_, Y_, rms_per_test, per_max_test, per_min_test, mae_test = sess.run(
[cost, print_hypo, print_Y, rms_per, per_max, per_min, mae], feed_dict={X: x_P, Y: y_P})#, phase : 0})

cost_train, hy_val, rms_per_train, per_max_train, per_min_train, mae_train = sess.run(
[cost, print_hypo, rms_per, per_max, per_min, mae], feed_dict = {X: x_data, Y: y_data})#, phase : 1})

for i in range(1):
       out = open('M1NR_1batch_0.0001_9L_64W_reg_off_residual_def_test2_diff_%d.out' %(step) ,'a')
       out.write('hypo %d : %e PF %d : %e\n'%(i,hypo_[i],i,Y_[i]))
       out.close()
		
for i in range(1):
       print('hypo %d : %e PF %d : %e\n'%(i,hypo_[i],i,Y_[i]))


print(' cost_train', cost_train, "cost_test ", cost_test, per_max_test, per_min_test, rms_per_test)
print(' hy_train :', hy_val)
print(' hy_test :', hypo_)
print("train: cost    per_max     per_min     rms_per    \n    [", cost_train, per_max_train, per_min_train, rms_per_train,"]")
print("test : cost    per_max     per_min     rms_per    \n    [", cost_test, per_max_test, per_min_test, rms_per_test,"]")
#out = open('CNN_opt_test_2batch_0.0001.out','a')
#out.write('time elapsed:  {:.2f}s \n'.format(time.time() - start_time))
#out.close ()
