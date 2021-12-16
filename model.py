from tensorflow import keras
from keras.layers import Flatten

import config
from main import load_dataset

inp, out = load_dataset()

print(inp.shape)
print(out.shape)

model = keras.Sequential()
model.add(keras.Input(shape=(24*config.NUM_MFCC)))

# first parameter is number of nodes. activation is the activation function
model.add(keras.layers.Dense(13, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]

model.fit(inp, out, epochs=150, batch_size=10)

loss, accuracy = model.evaluate(inp, out)
print('Accuracy: %.2f' % (accuracy*100))

predictions = (model.predict(inp) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %s (expected %d)' % (inp[i].tolist(), predictions[i], out[i]))

# how does the input work? 
# I now I'll have like 95 ndarrays (95 windows) of 13 elements each (13 coefficients for each window)
# will I run classification on each of the 95 arrays then do something?
# or input the whole 95 ndarrays and classify?

# speaking of https://bibliotecadigital.ipb.pt/bitstream/10198/20502/1/pauta-relatorio-43.pdf
# we input the whole 95 ndarrays, but only after we turn them into a single (unidimensional) array.
# they use tf.keras.layers.Conv1D instead of sequential?

# it is important to note that I'll need to do some trial and error to find the best
# number of layers and number of neurons. check page 52 of the above link to get some insight.

# remember to make/do dropout to avoid overfitting (I believe this is stopping the training at certain point)

# something I'll need to do: divide the dataset into training and validation datasets. training > validation
# and of course, divide them by result (pathological or not). (done this already) 