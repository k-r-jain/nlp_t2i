import bcolz
import numpy as np
import pickle
glove_path = '/home/kartik/Desktop'


# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

# with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
    
# vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

print(glove['<unk>'])