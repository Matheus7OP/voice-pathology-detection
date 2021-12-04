from tensorflow import keras

def main():
    model = keras.Sequential()

    model.add(keras.Dense(13, input_dim=8, activation='relu'))
    model.add(keras.Dense(8, activation='relu'))
    model.add(keras.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

    # something I'll need to do: divide the dataset into training and validation datasets. 
    # and of course, divide them by result (pathological or not). (done this already)
