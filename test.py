import tensorflow as tf
import numpy as np
import random
import time

#step = 1
a = 49000
batch_size = 49000
# = 63816 
#atch_size = 31908 
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
#keep_prob = tf.placeholder(tf.float32)

train = np.loadtxt('train_PF.csv', delimiter = ',', dtype = np.float32)
test = np.loadtxt('test_PF.csv', delimiter = ',', dtype = np.float32)
P = np.loadtxt('merge_print_PF.csv', delimiter = ',', dtype = np.float32)

x_data = train[1:49001,1:-1]
x_data = x_data.reshape([-1,8,8,4])
#x_data = x_data.reshape(-1,8,8,4)
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
#x_data=x_data.reshape([-1,192])
x_data = x_data.reshape([-1,256])

x_test = test[1:7001,1:-1]
x_test = x_test.reshape([-1,8,8,4])
#x_test = x_test.reshape(-1,8,8,4)
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
#x_data_tf=tf.reshape(x_data, [-1,192])
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
X = tf.placeholder(tf.float32, [None, 256])
X_img = tf.reshape(X, [-1, 8, 8, 4])
Y = tf.placeholder(tf.float32, [None, 1])
#Max_Y = tf.placeholder(tf.float32)
#Min_Y = tf.placeholder(tf.float32)

# L1 ImgIn shape=(?, 8, 8, 3)
W1 = tf.Variable(tf.random_normal([3, 3, 4, 64], stddev=0.01), name = 'pf_w1')
#    Conv     -> (?, 8, 8, 10)
#    Pool     -> (?, 4, 4, 10)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv1')
L1 = tf.nn.relu(L1, name = 'pf_relu1')

W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), name = 'pf_w2')
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv2')
L2 = tf.nn.relu(L2, name = 'pf_relu2')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), name = 'pf_w3')
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv3')
L3 = tf.nn.relu(L3, name = 'pf_relu3')

#L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), name = 'pf_w4')
#    Conv     -> (?, 8, 8, 10)
#    Pool     -> (?, 4, 4, 10)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv4')
L4 = tf.nn.relu(L4, name = 'pf_relu4')

L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pf_pool1')

W5 = tf.Variable(tf.random_normal([3, 3, 64,128], stddev=0.01), name = 'pf_w5')
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv5')
L5 = tf.nn.relu(L5, name = 'pf_relu5')

W6 = tf.Variable(tf.random_normal([3, 3,128,128], stddev=0.01), name = 'pf_w6')
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv6')
L6 = tf.nn.relu(L6, name = 'pf_relu6')

W7 = tf.Variable(tf.random_normal([3, 3,128,128], stddev=0.01), name = 'pf_w7')
L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv7')
L7 = tf.nn.relu(L7, name = 'pf_relu7')

W8 = tf.Variable(tf.random_normal([3, 3,128,128], stddev=0.01), name = 'pf_w8')
L8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv8')
L8 = tf.nn.relu(L8, name = 'pf_relu8')

W9 = tf.Variable(tf.random_normal([3, 3,128,128], stddev=0.01), name = 'pf_w9')
L9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME', name = 'pf_conv9')
L9 = tf.nn.relu(L9, name = 'pf_relu9')

L9 = tf.nn.max_pool(L9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pf_pool2')
#L6 = tf.nn.dropout(L6, keep_prob=keep_prob)
L9_flat = tf.reshape(L9, [-1,2*2*128], name = 'pf_flat')

# Final FC 2x2x90 inputs -> 1 outputs
W10 = tf.get_variable("pf_w10", shape=[2*2*128,256], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([256]), name = 'pf_b10')
L10 = tf.nn.relu(tf.matmul(L9_flat, W10) + b10, name = 'pf_relu10')
#L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

W11 = tf.get_variable("pf_w11", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
b11 = tf.Variable(tf.random_normal([128]), name = 'pf_b11')
L11 = tf.nn.relu(tf.matmul(L10, W11) + b11, name = 'pf_relu11')
#L8 = tf.nn.dropout(L8, keep_prob=keep_prob)

W12 = tf.get_variable("pf_w12", shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
b12 = tf.Variable(tf.random_normal([64]), name = 'pf_b12')
L12 = tf.nn.relu(tf.matmul(L11, W12) + b12, name = 'pf_relu12')
#L9 = tf.nn.dropout(L9, keep_prob=keep_prob)

W13 = tf.get_variable("pf_w13", shape=[64,1], initializer=tf.contrib.layers.xavier_initializer())
b13 = tf.Variable(tf.random_normal([1]), name = 'pf_b13')

hypothesis = tf.matmul(L12, W13) + b13
regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) +\
               tf.nn.l2_loss(W6) + tf.nn.l2_loss(W7) + tf.nn.l2_loss(W8) + tf.nn.l2_loss(W9) + tf.nn.l2_loss(W10) +\
               tf.nn.l2_loss(W11) + tf.nn.l2_loss(W12) + tf.nn.l2_loss(W13)
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
rms_per = tf.sqrt(tf.reduce_mean(tf.square(per)))
per_max = tf.reduce_max(per)
per_min = tf.reduce_min(per)

# Launch the graph in a session.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.Session()#config=config)
# Initializes global variables in the graph.
start_time = time.time()
sess.run(tf.global_variables_initializer())
# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver(max_to_keep=30)

out = open('M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev.out','w')
#out=open('ML_9layer.out','a')
#out.write('#-----------------------------------------------------------------------------------\n')
#out.write('#-----------------------------------------------------------------------------------\n')
#out.write('#-----------------------------------------------------------------------------------\n')
out.close ()

a=np.logspace(1, 4, 100, endpoint=True, dtype=int)
b=np.logspace(4, 6, 200, endpoint=True, dtype=int)
c=np.logspace(6, 6.30103, 41, endpoint=True, dtype=int)

stop_count = 0
stop_cost = 20

for epoch in range(50000):
    step = epoch
    
    p = np.random.permutation(len(x_data))
    x_data=x_data[p]
    y_data=y_data[p]
    total_batch = 1
    avg_cost_test = avg_cost_train = 0

    for i in range(total_batch):
        cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, optimizer], feed_dict = {X: x_data[i*batch_size:(i+1)*batch_size], Y: y_data[i*batch_size:(i+1)*batch_size]})
    if epoch % 10 == 0:
        rms_per_train = sess.run(rms_per, feed_dict={X: x_data, Y: y_data})
        
        cost_test, rms_test, diff_max_test, diff_min_test, rms_per_test, per_max_test, per_min_test = sess.run(
        [cost, rms, diff_max, diff_min, rms_per, per_max, per_min], feed_dict={X: x_test, Y: y_test})

        avg_cost_train += cost_val / total_batch
        avg_cost_test += cost_test / total_batch
        
                       
        out = open('M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev.out','a')
        print(epoch,'epoch' , ' cost_train', avg_cost_train, "cost_test ", avg_cost_test, per_max_test, per_min_test, rms_per_test, rms_per_train)
        out.write('%d Cost_train %e  Cost_test %e %e %e %e %e \n' %(epoch,cost_val,cost_test,per_max_test,per_min_test,rms_per_test, rms_per_train))
        out.close ()
        saver.save(sess, 'model/M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev/M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev', global_step=epoch, write_meta_graph=False)
        
        if stop_cost >= cost_test:
            stop_cost = cost_test
            stop_count = 0
        else:
            stop_count += 1

        if stop_count == 20:
            break
        
print('Trained Model Saved.')
print("train: cost    per_max     per_min     rms_per    \n    ", sess.run([cost, per_max, per_min, rms_per], feed_dict={X: x_data, Y: y_data}))

                                                                                                                                           
print("test : cost    per_max     per_min     rms_per    \n    ", sess.run([cost, per_max, per_min, rms_per], feed_dict={X: x_test, Y: y_test}))
print("time elapsed: {:.2f}s".format(time.time() - start_time))
print_hypo = hypothesis * (max_y - min_y) + min_y
print_Y = Y * (max_y - min_y) + min_y

saver.restore(sess, 'model/M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev/M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev-%d' %(step))
out = open('M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev_diff_%d.out' %(step),'w')

cost_test, hypo_, Y_, rms_per_test, per_max_test, per_min_test = sess.run(
[cost, print_hypo, print_Y, rms_per, per_max, per_min], feed_dict={X: x_P, Y: y_P})

cost_val, hy_val, rms_per_train, per_max_train, per_min_train = sess.run(
[cost, print_hypo, rms_per, per_max, per_min], feed_dict = {X: x_data, Y: y_data})

for i in range(1):
       out = open('M1NR_1batch_0.0001_9L_64W_reg_off_FC_rev_diff_%d.out' %(step) ,'a')
       out.write('hypo %d : %e PF %d : %e\n'%(i,hypo_[i],i,Y_[i]))
       out.close()
		
for i in range(1):
       print('hypo %d : %e PF %d : %e\n'%(i,hypo_[i],i,Y_[i]))


print(' cost_train', cost_val, "cost_test ", cost_test, per_max_test, per_min_test, rms_per_test)
print(' hy_train :', hy_val)
print(' hy_test :', hypo_)
print("train: cost    per_max     per_min     rms_per    \n    [", cost_val, per_max_train, per_min_train, rms_per_train,"]")
print("test : cost    per_max     per_min     rms_per    \n    [", cost_test, per_max_test, per_min_test, rms_per_test,"]")
#out = open('CNN_opt_test_2batch_0.0001.out','a')
#out.write('time elapsed:  {:.2f}s \n'.format(time.time() - start_time))
#out.close ()
