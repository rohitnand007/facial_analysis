# from keras.models import Sequential
# from keras.layers import Activation
# from keras.optimizers import SGD
# from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adadelta
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
import keras

def test_method():
	print("++++++++++++++++++++++++++++++++++++++++++++Printing LOG line++++++++++++++++++++++++")

def define_model1(trainData, trainLabels, testData, testLabels, output_model_path):
	# define the architecture of the network
	print("[INFO] constructing model with the kaggle dataset......")
	model = Sequential()
	model.add(Dense(1536, input_dim=2304, init="uniform",
		activation="relu"))
	model.add(Dense(786, activation="relu", kernel_initializer="uniform"))
	model.add(Dense(7))
	model.add(Activation("softmax"))

	# train the model using SGD
	print("[INFO] compiling model...")
	sgd = SGD(lr=0.01)
	model.compile(loss="categorical_crossentropy", optimizer=sgd,
		metrics=["accuracy"])
	model.fit(trainData, trainLabels, epochs=100, batch_size=64,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating on testing set...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
		accuracy * 100))
	# dump the network architecture and weights to file
	print("[INFO] dumping architecture and weights to file...")
	model.save(output_model_path)
	print("[INFO] process completed")

def define_model2(trainData, trainLabels, testData, testLabels, output_model_path):
	# define the architecture of the network
	print("[INFO] constructing model with the kaggle dataset......")
	img_rows, img_cols = 48, 48
	model = Sequential()
	model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(img_rows, img_cols, 3)))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(128, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(128, 3, 3))
	model.add(PReLU(init='zero', weights=None))

	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))

	model.add(Dense(7))
	model.add(Activation("softmax"))

	# train the model using SGD
	print("[INFO] compiling model...")
	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',
	              optimizer=ada,
	              metrics=['accuracy'])
	model.fit(trainData, trainLabels, epochs=200, batch_size=64,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating on testing set...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=64, verbose=1)
	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
		accuracy * 100))
	# dump the network architecture and weights to file
	print("[INFO] dumping architecture and weights to file...")
	model.save(output_model_path)
	print("[INFO] process completed")	

def VGG_16(trainData, trainLabels, testData, testLabels, output_model_path, weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(48,48,3)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax'))

	# train the model using SGD
	print("[INFO] compiling model...")
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	model.fit(trainData, trainLabels, epochs=100, batch_size=64,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating on testing set...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=64, verbose=1)
	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
		accuracy * 100))
	# dump the network architecture and weights to file
	print("[INFO] dumping architecture and weights to file...")
	model.save(output_model_path)
	print("[INFO] process completed")		

