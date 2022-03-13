import numpy as np
from tensorflow.keras import Sequential, Input, layers

from main import load_dataset_with_features
from config import K_VALUE, MASK_VALUE, N_EPOCHS

inp, out = load_dataset_with_features()
# inp, out = load_dataset()

print(inp.shape)
print(out.shape)

rnd_permutation = np.random.RandomState().permutation(len(inp))
accuracies = list()

tpcs = list()
tncs = list()

fpcs = list()
fncs = list()

# cross-validation
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
    model.add(Input(shape=(len(training_in[0]), )))

    model.add(layers.Masking(mask_value=MASK_VALUE))

    # 1st parameter is the number of nodes. 2nd is the activation function
    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    # debug
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

    tpc, tnc = 0, 0
    fpc, fnc = 0, 0

    # calculating true/false positive/negative metrics
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
print(f"Mean true positive: {round(mean_tpc*100, 2)}%")
# print(f"TPs: {[(str(round(v*100, 2)) + '%') for v in tpcs]}")

mean_tnc = sum(tncs) / len(tncs)
print(f"Mean true negative: {round(mean_tnc*100, 2)}%")
# print(f"TNs: {[(str(round(v*100, 2)) + '%') for v in tncs]}")

mean_fpc = sum(fpcs) / len(fpcs)
print(f"Mean false positive: {round(mean_fpc*100, 2)}%")
# print(f"FPs: {[(str(round(v*100, 2)) + '%') for v in fpcs]}")

mean_fnc = sum(fncs) / len(fncs)
print(f"Mean false negative: {round(mean_fnc*100, 2)}%")
# print(f"FNs: {[(str(round(v*100, 2)) + '%') for v in fncs]}")

mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Accuracies: {[(str(round(v*100, 2)) + '%') for v in accuracies]}")
print(f"Mean accuracy: {round(mean_accuracy*100, 2)}%")
