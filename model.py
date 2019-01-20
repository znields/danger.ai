import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)

def y_data_get():
    
    y_list = []
    
    num_examples = 12
    
    rootDir = r"C:\Users\Mitchell\Documents\GitHub\attention.ai\openpose\y"
    
    for dirName, subDir, fileList in os.walk(rootDir, topdown=False):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)
            y_list.append(np.load(dirName + "\\" + fname))
            print(y_list[len(y_list)-1].shape)
    
    
    
    y_list_comp = []
    for i in range(num_examples):#(int)(len(y_list)/2)):
        y_list_comp.append(np.concatenate((y_list[i*2], y_list[i*2 + 1])))
        print(y_list_comp[i].shape)

    return y_list_comp


def x_data_get():
    #retrieve saved npy files and combine
    #them and put them into a list
    
    
    x_list = []
    
    num_examples = 12 #total number of vidoes, not npy files, assuming each video has 2 npy files
    
    rootDir = r"C:\Users\Mitchell\Documents\GitHub\attention.ai\openpose\x"
    
    for dirName, subDir, fileList in os.walk(rootDir, topdown=False):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)
            x_list.append(np.load(dirName + "\\" + fname).T)
            print(x_list[len(x_list)-1].shape)

    x_list_comp = []
    
    for i in range(num_examples):#(int)(len(y_list)/2)):
        x_list_comp.append(np.concatenate((x_list[i*2], x_list[i*2 + 1])))
        print(x_list_comp[i].shape)

    return x_list_comp
	
	
def x_data_reshape(x_list_comp):
    #assuming the data is in the form of a list
    #where each element in the list is a numpy
    #array of shape (1080,162) --need to transform
    #each element of the list into (72,15,162)
    #each element of the output will represent
    #the final training data for one video
    
    x_list_final = []
    
    for npy in x_list_comp:
        assert npy.shape == (1080,162)
        x_list_final.append(npy.reshape(72,15,162))
        
    return x_list_final
	
	
	

train_list = x_data_get()

train_x_list = x_data_reshape(train_list)

train_y_list = y_data_get()


train_x = np.concatenate(train_x_list, axis=0)
train_y = np.concatenate(train_y_list, axis=0)

#embedding_vecor_length = 98
model = Sequential()
#model.add(Embedding(5000, embedding_vecor_length, input_length=15))
model.add(LSTM(100, input_shape = (15,162)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_x_list[0], train_y_list[0], epochs=10, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(train_x, train_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

preds = model.predict(train_x)
np.save("predictions.npy", preds)