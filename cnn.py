from tensorflow.python.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import argparse
import tensorflow as tf
from data_model import data_model
import numpy as np
import os
from extract_features_and_extend import extract_features_and_extend

dm = data_model()
path = os.path.abspath(".")
train_data_path = path + "\\training_data_samples_6066_features_22_window_size_32.npy"
train_Y_data_path = path + "\\predictions.npy"

train_X = np.load(train_data_path)
#print(train_X)
#print(train_X[0][0].size)
train_Y = np.load(train_Y_data_path)
#train_Y = train_Y.reshape(-1, 1)
#print(train_Y.shape)

testing_data = dm.test(length=32,pred_length=24, its=10)
testing_metadata = testing_data[0]
testing_time_series = testing_data[1]
testing_data_predictions = testing_data[2]
print(type(testing_metadata))
#print(len(testing_data[0][0]))
#print(len(testing_data[2][0]))
arg = argparse.ArgumentParser()
arg.add_argument("Layers", type=int, help="input number of CNN layers")
arg.add_argument("--layers", action='store_true', required=True, help="input number of CNN layers")

args = arg.parse_args()
layer_num = args.Layers

model = models.Sequential()

# input = [num_features * num_stations]
# expected_vals = [num_stations]
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(170, 22)))
model.add(layers.MaxPooling1D(2))
for i in range(layer_num-1):
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(170, activation="linear"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(train_X, train_Y, epochs=10)

#print(preds)
print(train_Y.shape)
preds = []
#24 predictions
for i in range(24):
    #test_X = np.asarray(testing_metadata).astype(np.float32)
    test_X = extract_features_and_extend(testing_data[2], testing_metadata, preds)
    print(test_X.size)
    preds = model.predict(testing_metadata)
    print(preds.size)
def mse(n, pred, val):
    mse = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            mse += (pred[i][j] - val[i][j])**2
    return mse/n

train_error = mse(train_Y.shape[0], preds, train_Y)
print("Train error for CNN with {} layers: {}".format(layer_num, train_error))