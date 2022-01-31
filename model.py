import numpy as np
from tensorflow.keras import Sequential, Input, layers

from main import load_dataset_with_features, load_dataset
from config import K_VALUE, MASK_VALUE, N_EPOCHS

"""
Sendo assim, nesta pesquisa, o sinal de voz digitalizado passa a ser fracionado
em uma sequência de segmentos de mesmo comprimento. Cada um destes segmentos
será inserido no classificador (um por vez) para que o mesmo defina a qual
classe cada um dos segmentos apresentados pertencem. Ao fim da etapa de
classificação dos segmentos realiza-se a computação da classe predominante a
fim de definir em qual categoria o sinal de voz como um todo será classificado.
"""

# inp, out = load_dataset_with_features()
inp, out = load_dataset()

print(inp.shape)
print(out.shape)

# seed: 42
rnd_permutation = np.random.RandomState().permutation(len(inp))
accuracies = list()

tpcs = list()
tncs = list()

fpcs = list()
fncs = list()

for k in range(K_VALUE):
    fold_size = (len(inp) / K_VALUE)
    training_fold_bg = fold_size * k
    training_fold_ed = fold_size * (k+1)

    training_in = []
    training_out = []

    validation_in = []
    validation_out = []

    for idx_enum, i in enumerate(rnd_permutation):
        if idx_enum >= training_fold_bg and idx_enum < training_fold_ed:
            validation_in.append(inp[rnd_permutation[i]])
            validation_out = np.append(validation_out, out[rnd_permutation[i]])
        else:
            training_in.append(inp[rnd_permutation[i]])
            training_out = np.append(training_out, out[rnd_permutation[i]])

    training_in = np.asarray(training_in)
    validation_in = np.asarray(validation_in)

    model = Sequential()
    model.add(Input(shape=(len(training_in[0]), )))  # 436800 or 3562

    """
    [...] sendo analisados classificadores com 04, 05 e 06 camadas ocultas.
    A motivação de alterar a quantidade de camadas ocultas está relacionada
    com o interesse de avaliar os efeitos que a profundidade da DNN causam na
    eficiência do classificador.

    Durante as etapas de treino e validação, o sinal de voz é fracionado em
    segmentos estacionários de 16 ms, possuindo no total 400 amostras. A camada
    de entrada da DNN recebe as 400 amostras (instâncias) referentes ao
    segmento em análise. Neste trabalho, cada camada oculta dos classificadores
    é composta por 200 neurônios [...]
    """

    model.add(layers.Masking(mask_value=MASK_VALUE))

    # first parameter is number of nodes. activation is the activation function
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    # model.add(layers.Flatten())

    # model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['accuracy'])

    model.fit(training_in, training_out, epochs=N_EPOCHS, batch_size=10)

    _, accuracy = model.evaluate(validation_in, validation_out)
    print('Accuracy: %.2f' % (accuracy*100))
    accuracies.append(accuracy)

    predictions = (model.predict(validation_in) > 0.5).astype(int)

    tpc = 0
    tnc = 0

    fpc = 0
    fnc = 0

    for i in range(len(predictions)):
        if predictions[i] == 0:
            if predictions[i] == int(validation_out[i]):
                tnc += 1
            else:
                fpc += 1
        else:
            if predictions[i] == int(validation_out[i]):
                tpc += 1
            else:
                fnc += 1

    tpcs.append(tpc/float(len(predictions)))
    tncs.append(tnc/float(len(predictions)))

    fpcs.append(fpc/float(len(predictions)))
    fncs.append(fnc/float(len(predictions)))


mean_tpc = sum(tpcs) / len(tpcs)
# print(f"TPs: {[(str(round(v*100, 2)) + '%') for v in tpcs]}")
print(f"Mean true positive: {round(mean_tpc*100, 2)}%")

mean_tnc = sum(tncs) / len(tncs)
# print(f"TNs: {[(str(round(v*100, 2)) + '%') for v in tncs]}")
print(f"Mean true negative: {round(mean_tnc*100, 2)}%")

mean_fpc = sum(fpcs) / len(fpcs)
# print(f"FPs: {[(str(round(v*100, 2)) + '%') for v in fpcs]}")
print(f"Mean false positive: {round(mean_fpc*100, 2)}%")

mean_fnc = sum(fncs) / len(fncs)
# print(f"FNs: {[(str(round(v*100, 2)) + '%') for v in fncs]}")
print(f"Mean false negative: {round(mean_fnc*100, 2)}%")

mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Accuracies: {[(str(round(v*100, 2)) + '%') for v in accuracies]}")
print(f"Mean accuracy: {round(mean_accuracy*100, 2)}%")

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
"""
