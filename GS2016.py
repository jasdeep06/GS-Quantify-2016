import tensorflow as tf

x=tf.placeholder(tf.float32,shape=[None,16])
y_=tf.placeholder(tf.float32,shape=[None,16])
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def read_from_csv(filename_queue):
    reader=tf.TextLineReader()
    key,value=reader.read(filename_queue)
    record_defaults=[[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.],[1.],[1.]]
    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17=tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16])
    labels=tf.pack([col17])
    return features,labels
def input_pipeline(filenames,batch_size,num_epochs=None):
    filename_queue=tf.train.string_input_producer([filenames],num_epochs=num_epochs,shuffle=True)
    features,labels=read_from_csv(filename_queue)
    min_after_dequeue=100
    capacity=300
    feature_batch,label_batch=tf.train.shuffle_batch([features,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
    return feature_batch,label_batch

x,y_=input_pipeline('csvnew1.csv',20,300)
#input layer
W_1=weight_variable([16,16])
b_1=bias_variable([16])
y_1=tf.nn.relu(tf.matmul(x,W_1)+b_1)

#hidden layer
W_2=weight_variable([16,16])
b_2=bias_variable([16])
y_2=tf.nn.softmax(tf.matmul(y_1,W_2)+b_2)


cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_2),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.00000001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_2,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
summary_cross=tf.scalar_summary('cost',cross_entropy)
summaries = tf.merge_all_summaries()

init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session()
summary_writer = tf.train.SummaryWriter('stats', sess.graph)


# Initialize the variables (like the epoch counter).
sess.run(init_op)
sess.run(tf.initialize_local_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

count=0
try:
    while not coord.should_stop():

        #print("training....")
        summary_writer.add_summary(sess.run(summaries), count)



        sess.run(train_step)
        if count in range(300,90000,300):
            print(sess.run(cross_entropy))
        count=count+1
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
