# Building a 4-Gram Predictor in Tensorflow

So, a basic Natural language excersize. We split a corpus (bunch of words from somewhere) into 4-grams (4-tuples of words in the order they appear in a text). The model's purpose is to use the first three words in the tuple to predict the last.

	four_tuple = ['cat','in','the','hat']
	prediction = model(four_tuple[:3])
	prediction == 'hat'

Lots of basic learning involved here.

### Preprocessing the Data

All words are stored in a list that is used to encode each word to an int. IE. 'cat' -> 20.

		['cat','in','the','hat'] -> [20, 101, 2, 40]
		
Then, each number is converted to it's one-hot representation. If there were a total vocabulary size of 5, the on-hot representation of two is:

		2 -> [0,0,1,0,0]
		
### The Model and Results

Model MK1 simply used an input and output fully connected, activated by softmax and minimizing cross-entropy.  

	x:[22,33,11]	y:[0,0,0...,0,0,0] 
	
where x is the slice corresponding to the first three word ids of a four-gram and y is the one-hot representation of the last word.

This worked terribly. A little investigation showed that the model was simply predicting that the fourth token in the four gram would be a '.' for all four-grams. Clever, but stupid.

So skipping a few MKs, MK2, the best approach so far is

	shape(x):3*len(vocab)	shape(y): len(vocab)

which is meant to indicate that x is the flattened list of the one-hot reps of the first 3 words and y remains the one-hot of the last word.

This approach resulted in a peak accuracy of 0.31 much better than the previous peak of 0.14. Investigation of the predictions being made showed there to be a wide variety as well, not fixated on one specific prediction. 

MK3 with a hidden layer with equal dims to the input layer gives us a peak accuracy of 0.33. Is this good? I'm not sure. I need a benchmark.
According to [some slides on the internet](http://www.coling-2014.org/COLING%202014%20Tutorial-fix%20-%20Tomas%20Mikolov.pdf) above 0.1 is actually a pretty decent showing. Notably superceded by the CBOW architecture. I'm going to attribute the relatively high accuracy to the extremely limited vocabulary (250 words). This also somewhat assuages my concerns considering Mnist Models get accuracies in the range of 90+%. Of course this makes sense since humans are close to faultless at recognizing (semi-neat) hand-writing, but I dare you to try and guess what the next word's going to- potato.

### The Code

The largest diffifculty of learning to work with tensorflow is coming to understand that nothing is immediately evaluated, everything is just a structure for when the model eventually runs. This is disorienting and presents a decent curve, especially if you have little to no experience with the Machine Learning Theory behind Neural Nets.

So a couple things learned:

	x = tf.placeholder(tf.float32,shape=[None,in_space])

x is our input placehoder variable. It's helpful for me to conceptualize it as the input argument for a function. It has no value yet, so we have to be patient to see what's actually going on.

	y = tf.nn.softmax(tf.matmul(r,W)+b)
	
y is our output tensor. In a drawing of an ANN, it'd be represented as the output neurons. A thing to notice is that all relationships described with softmax/ReLu/some other activation are our neurons/nodes. These are the dots of the graph.

	W = tf.Variable(tf.random_normal([in_space,class_space]))
	b = tf.Variable(tf.zeros([class_space]))
	
And the weights (W) are the lines connecting the dots, they're the connections. The biases favour certain nodes and not others. 

To connect more layers into the structure, we add more weights and biases and associate them to activation functions (nodes).

[Mostly Based off the tensorflow mnist example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py)

[Code Repo](https://github.com/KaroAntonio/4-Gram-Tensorflow)