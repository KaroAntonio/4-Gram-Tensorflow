import tensorflow as tf

from prep_sentences import SentenceData
import input_data

print('Load Data..')
data = SentenceData('rawSentences.txt',one_hot=True)
sess = tf.InteractiveSession()

#MODEL
#here the None paramater indicates the first dimension of input (batch size), may be of any size. len(data) is vocab size
in_space = data.x_dim
rep_dims = data.x_dim
num_hidden = 256
class_space = len(data)

#X
x = tf.placeholder(tf.float32,shape=[None,in_space]) #input placeholder variable

#X->REP
W_xr = tf.Variable(tf.random_normal([in_space,rep_dims]))
b_xr = tf.Variable(tf.zeros([rep_dims]))

#REP
r = tf.nn.softmax(tf.matmul(x, W_xr) + b_xr)

#REP->HIDDEN
W_rh = tf.Variable(tf.random_normal([rep_dims,num_hidden]))
b_rh = tf.Variable(tf.zeros([num_hidden]))

#HIDDEN
h = tf.nn.softmax(tf.matmul(r, W_rh) + b_rh)

#HIDDEN->Y
#W_hy = tf.Variable(tf.random_normal([num_hidden,class_space]))
W_hy = tf.Variable(tf.random_normal([rep_dims,class_space]))
b_hy = tf.Variable(tf.zeros([class_space]))

#Y
y = tf.nn.softmax(tf.matmul(r,W_hy)+b_hy)

#place holder for the correct labels
y_ = tf.placeholder(tf.float32,shape=[None,class_space]) 

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

learning_rate = 0.01
report_interval = 1000
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #becomes an array.. [F,T,F]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #[F,T,F]->sum([0,1,0])/len ... something like that

print('Training...')
for i in range(100000):
	batch_xs, batch_ys = data.next(50)
	train_step.run({x: batch_xs, y_: batch_ys})

	#Test trained model...
	if i%report_interval == 0:
		prediction = tf.argmax(y,1) #gets index of greatest element in tensor
		probabilities = y
		test_xs, test_ys = data.next(50)
		#print(cross_entropy)
		#print('pre',prediction.eval({x:test_xs}))
		#print('probs',probabilities.eval({x:test_xs}))
		print('accuracy:',accuracy.eval({x:test_xs, y_:test_ys}))
		#print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
	

