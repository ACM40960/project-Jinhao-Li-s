import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def cnn_inference(images, keep_prob):
    #Convolution kernels
    W_conv = {
        #Generate normally distributed random numbers with excessive deviations removed, with a standard deviation of 0.1
        'conv1': tf.Variable(tf.random.truncated_normal([3, 3, 3, 32],
                                                        stddev=0.1)),
        'conv2': tf.Variable(tf.random.truncated_normal([3, 3, 32, 32],
                                                        stddev=0.1)),
        'conv3': tf.Variable(tf.random.truncated_normal([3, 3, 32, 64],
                                                        stddev=0.1)),
        'conv4': tf.Variable(tf.random.truncated_normal([3, 3, 64, 64],
                                                        stddev=0.1)),
        'conv5': tf.Variable(tf.random.truncated_normal([3, 3, 64, 128],
                                                        stddev=0.1)),
        'conv6': tf.Variable(tf.random.truncated_normal([3, 3, 128, 128],
                                                        stddev=0.1)),
        'fc1_1': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_2': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_3': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_4': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_5': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_6': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        'fc1_7': tf.Variable(tf.random.truncated_normal([5*30*128, 65],
                                                        stddev=0.01)),
        } 
    #Bias 
    b_conv = { 
        'conv1': tf.Variable(tf.constant(0.1, dtype=tf.float32, 
                                         shape=[32])),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[32])),
        'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[64])),
        'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[64])),
        'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[128])),
        'conv6': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[128])),
        'fc1_1': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_2': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_3': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_4': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_5': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_6': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        'fc1_7': tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[65])),
        } 


    # The first convolutional layer with strides() is the step size of each dimension
    conv1 = tf.nn.conv2d(images, W_conv['conv1'], strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    conv1 = tf.nn.relu(conv1)
 
    # The second convolutional layer
    conv2 = tf.nn.conv2d(conv1, W_conv['conv2'], strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    conv2 = tf.nn.relu(conv2)
    # In layer 1, the pooling layer, Max represents the maximum output in the rectangular neighborhood, and ksize represents the pooling window size
    pool1 = tf.nn.max_pool2d(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
 
    # The third convolutional layer
    conv3 = tf.nn.conv2d(pool1, W_conv['conv3'], strides=[1,1,1,1], padding='VALID')
    conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    conv3 = tf.nn.relu(conv3)
 
    # The fourth convolutional layer
    conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1,1,1,1], padding='VALID')
    conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
    conv4 = tf.nn.relu(conv4)
    # Layer 2 pooling layer
    pool2 = tf.nn.max_pool2d(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # The fifth convolutional layer
    conv5 = tf.nn.conv2d(pool2, W_conv['conv5'], strides=[1,1,1,1], padding='VALID')
    conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
    conv5 = tf.nn.relu(conv5)

    # The sixth convolutional layer
    conv6 = tf.nn.conv2d(conv5, W_conv['conv6'], strides=[1,1,1,1], padding='VALID')
    conv6 = tf.nn.bias_add(conv6, b_conv['conv6'])
    conv6 = tf.nn.relu(conv6)
    # Layer 3 pooling layer
    pool3 = tf.nn.max_pool2d(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
 
    #Layer 1_1 Fully connected layer
    reshape = tf.reshape(pool3, [-1, 5 * 30 * 128])
    #Let each neuron stop working with a certain probability (keep_prob) to prevent overfitting
    fc1 = tf.nn.dropout(reshape, keep_prob)
    fc1_1 = tf.add(tf.matmul(fc1, W_conv['fc1_1']), b_conv['fc1_1'])
    
    #Layer 1_2 Fully connected layer
    fc1_2 = tf.add(tf.matmul(fc1, W_conv['fc1_2']), b_conv['fc1_2'])

    #Layer 1_3 Fully connected layer
    fc1_3 = tf.add(tf.matmul(fc1, W_conv['fc1_3']), b_conv['fc1_3'])

    #Layer 1_4 Fully connected layer
    fc1_4 = tf.add(tf.matmul(fc1, W_conv['fc1_4']), b_conv['fc1_4'])
    
    #Layer 1_5 Fully connected layer
    fc1_5 = tf.add(tf.matmul(fc1, W_conv['fc1_5']), b_conv['fc1_5'])
    
    #Layer 1_6 Fully connected layer
    fc1_6 = tf.add(tf.matmul(fc1, W_conv['fc1_6']), b_conv['fc1_6'])
    
    #Layer 1_7 Fully connected layer
    fc1_7 = tf.add(tf.matmul(fc1, W_conv['fc1_7']), b_conv['fc1_7'])
   
    return fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, fc1_6, fc1_7


def calc_loss(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit1, labels=labels[:, 0]))
    #Save all summaries to disk for Tensorboard to display
    tf.compat.v1.summary.scalar('loss1', loss1)

    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit2, labels=labels[:, 1]))
    tf.compat.v1.summary.scalar('loss2', loss2)

    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit3, labels=labels[:, 2]))
    tf.compat.v1.summary.scalar('loss3', loss3)

    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit4, labels=labels[:, 3]))
    tf.compat.v1.summary.scalar('loss4', loss4)

    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit5, labels=labels[:, 4]))
    tf.compat.v1.summary.scalar('loss5', loss5)

    loss6 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit6, labels=labels[:, 5]))
    tf.compat.v1.summary.scalar('loss6', loss6)

    loss7 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit7, labels=labels[:, 6]))
    tf.compat.v1.summary.scalar('loss7', loss7)

    return loss1, loss2, loss3, loss4, loss5, loss6, loss7

#optimizer
def train_step(loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate):
    optimizer1 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op1 = optimizer1.minimize(loss1)

    optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op2 = optimizer2.minimize(loss2)

    optimizer3 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op3 = optimizer3.minimize(loss3)

    optimizer4 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op4 = optimizer4.minimize(loss4)

    optimizer5 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op5 = optimizer5.minimize(loss5)

    optimizer6 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op6 = optimizer6.minimize(loss6)

    optimizer7 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op7 = optimizer7.minimize(loss7)

    return train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7
   

def pred_model(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    labels = tf.reshape(tf.transpose(labels), [-1])
    logits = tf.concat([logit1, logit2, logit3, logit4, logit5, logit6, logit7], 0)
    prediction = tf.nn.in_top_k(logits, labels, 1)
    #Tf.cast (x,y) converts x to type Y
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', accuracy)
    return accuracy

