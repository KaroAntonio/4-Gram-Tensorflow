#import nltk
#process rawSentences.txt into 
#vocab->
#map of words to num occurances

class SentenceData():

	def __init__(self,fid,one_hot=False):
		self.one_hot = one_hot
		self.batch_ptr = 0 #points to the current batch
		self.tokens = self._buildTokens(fid)
		self.vocab = list(set(self.tokens))  
		self.four_grams = self._process()
		#set dimensions of x
		self.x_dim = len(self)*3 if one_hot else 3

	def _buildTokens(self,fid):
		#readlines from file, remove newlines, split into tokens
		tokens = []
		for l in open(fid):
			words = l.strip().split()
			for w in words:
				tokens.append(w.lower())
		return tokens
		
	def _process(self):
		#build 4-grams
		#build dictionaries
		four_grams = []
		for i in range(len(self.tokens)):
			w = self.tokens[i].lower()
			if i < len(self.tokens)-5:
				#4-gram tuples are floats [0,len(vocab)-1] that correspond to a word in vocab
				four_grams.append(tuple([self.wid(self.tokens[i+j]) for j in range (4)]))
		
		return four_grams

	def wid(self,w):
		'''
		(str)->(int)
		w: word to get the id of
		returns the id of the input word according to the vocab
		'''
		return self.vocab.index(w)

	def word(self,wid):
		'''(int)-> (str)
		wid: the id of the word to be returned
		return the word at index wid in vocab
		'''
		return self.vocab[wid]
	
	def next(self,n):
		'''()->(list of tuples)
		return the next batch of size n 
		'''
		batch_x = []
		batch_y = []
		for i in range(n):
			f_gram = self._get_gram(self.batch_ptr)
			y_i = self.wid_to_hot([f_gram[3]])[0]
			#first 3 words are input,
			#scaled to (0,1)
			#target is one-hot rep of last word
			if self.one_hot:
				one_hots = self.wid_to_hot(f_gram[:3])
				x_i = [e for oh in one_hots for e in oh]
			else:
				x_i = tuple([wid/len(self)for wid in f_gram[:3]])
			batch_x.append(x_i)	
			batch_y.append(y_i)
			self._inc_ptr()
		return(batch_x, batch_y)
	
	def _get_gram(self,i):
		return self.four_grams[i]

	def wid_to_hot(self, wids):
		'''([int])->([int])
		wids: word ids
		convert wid to a word's one-hot rep
		'''
		one_hots = []
		for wid in wids:
			oh = ([0]*len(self.vocab))
			oh[wid]=1
			one_hots.append(oh)
		return one_hots

	def _inc_ptr(self):
		self.batch_ptr = (self.batch_ptr + 1)%len(self.four_grams)
	def __len__(self):
		'''
		return size of vocab, that seems reasonable
		'''
		return len(self.vocab)

if __name__ == '__main__':
	data = SentenceData('rawSentences.txt',one_hot=True)
	print(data.wid('i'),data.word(200))
	test = data.four_grams[1]
	print(test, [data.word(wid) for wid in test])
	#test batches
	data.batch_ptr = len(data.four_grams)-1
	print(data.next(1),data.next(1), data.four_grams[-1], data.four_grams[:2])
		
