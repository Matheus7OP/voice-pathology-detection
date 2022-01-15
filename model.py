import numpy
from tensorflow import keras

import config
from main import load_dataset

inp, out = load_dataset()

print(inp.shape)
print(out.shape)

training_in = []
training_out = []

validation_in = []
validation_out = []

rnd_permutation = numpy.random.permutation(len(inp))

for i in rnd_permutation[:450]:
    training_in.append(inp[rnd_permutation[i]])
    training_out = numpy.append(training_out, out[rnd_permutation[i]])

training_in = numpy.asarray(training_in)

# print(training_in.shape)
# print(training_out.shape)

for i in rnd_permutation[450:]:
    validation_in.append(inp[rnd_permutation[i]])
    validation_out = numpy.append(validation_out, out[rnd_permutation[i]])

validation_in = numpy.asarray(validation_in)

model = keras.Sequential()
model.add(keras.Input(shape=(24*config.NUM_MFCC)))

"""
[...] sendo analisados classificadores com 04, 05 e 06 camadas ocultas.
A motivação de alterar a quantidade de camadas ocultas está relacionada
com o interesse de avaliar os efeitos que a profundidade da DNN causam na
eficiência do classificador.

Durante as etapas de treino e validação, o sinal de voz é fracionado em
segmentos estacionários de 16 ms, possuindo no total 400 amostras. A camada
de entrada da DNN recebe as 400 amostras (instâncias) referentes ao segmento
em análise. Neste trabalho, cada camada oculta dos classificadores é composta
por 200 neurônios [...]
"""

# first parameter is number of nodes. activation is the activation function
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]

model.fit(training_in, training_out, epochs=200, batch_size=10)

loss, accuracy = model.evaluate(validation_in, validation_out)
print('Accuracy: %.2f' % (accuracy*100))

# predictions = (model.predict(validation_in) > 0.5).astype(int)
# for i in range(5):
#     print('%s => %s (expected %d)' % (
#         inp[i].tolist(),
#         predictions[i],
#         validation_out[i]))

"""
how does the input work?
I now I'll have like 95 ndarrays (95 windows) of 13 elements each (13
coefficients for each window) will I run classification on each of the 95
arrays then do something? or input the whole 95 ndarrays and classify?

speaking of
https://bibliotecadigital.ipb.pt/bitstream/10198/20502/1/pauta-relatorio-43.pdf
we input the whole 95 ndarrays, but only after we turn them into a single
(unidimensional) array. they use tf.keras.layers.Conv1D instead of sequential?

it is important to note that I'll need to do some trial and error to find the
best number of layers and number of neurons. check page 52 of the above link
to get some insight.

remember to make/do dropout to avoid overfitting (I believe this is stopping
the training at certain point)

something I'll need to do: divide the dataset into training and validation
datasets. training > validation and of course, divide them by result
(pathological or not). (done this already)
"""
