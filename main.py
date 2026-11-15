import os

import numpy as np
from sklearn.impute import SimpleImputer

os.environ["KERAS_BACKEND"] = "jax"
import pandas as pd
from keras import layers, Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from matplotlib import pyplot as plt


TARGET = "quality_of_sleep"
entry_col = {"gender", "age", "sleep_duration", "physical_activity_level", "stress_level"}
target_col = {"quality_bad", "quality_fair", "quality_good"}

SEED = 31
STEP = 10
EPOCH = 500


def quality_class(value: int, lb: int, ub: int) -> int:
    return int(lb <= value <= ub)

def revert_quality_class(qualities: np.typing.NDArray):
    reverted = np.array([0, 0, 0])
    max_i = 0
    for i in range(len(qualities)):
        if qualities[i] > qualities[max_i ]:
            max_i = i
    reverted[max_i] = 1
    return reverted


def read_data():
    raw_data = pd.read_csv("./sleep_dataset.csv")
    s_data = pd.DataFrame()

    for col, col_data in raw_data.items():
        col_name = str(col).replace(" ", "_").lower()
        if col_name in entry_col or col_name == TARGET:
            s_data[col_name] = col_data
    s_data["gender"] = s_data["gender"].apply(lambda x: int(x == "Male"))
    s_data["quality_bad"] = s_data[TARGET].apply(quality_class, lb=0, ub=4)
    s_data["quality_fair"] = s_data[TARGET].apply(quality_class, lb=5, ub=7)
    s_data["quality_good"] = s_data[TARGET].apply(quality_class, lb=8, ub=10)

    #s_data[TARGET] = s_data[TARGET].apply(lambda x: int(x) / 10)

    pre = s_data.loc[:, [*entry_col]].values
    klas = s_data.loc[:, [*target_col]].values

    return pre, klas

def train(epoch: int = EPOCH):
    x, y = read_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED
    )

    mid_layers = [
        layers.Dense(4, activation="relu", input_dim=5),
        layers.Dense(4, activation="tanh"),
        layers.Dense(3, activation="softmax"),
    ]

    model = Sequential()

    for layer in mid_layers:
        model.add(layer)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=10, epochs=epoch, steps_per_epoch=STEP)

    prev: np.typing.NDArray = model.predict(x_test)
    prev_convert = np.array([revert_quality_class(p) for p in prev])

    precision = accuracy_score(y_test, prev_convert)
    print("\n----\n")
    print(f"Epochs: {EPOCH} | Step por Epoch: {STEP} |")
    print("\nPrecisão: ", precision)
    print("\nMatriz Confusão")

    for i in range(0, 3):
        c_mat = confusion_matrix(y_test[:, i], prev_convert[:, i])
        print("\n", c_mat)

    return precision


def graph():
    e = 50
    x = []
    y = []
    while e <= 500:
        p_sum = 0
        for i in range(0, 5):
            p_sum += train(e)
        x.append(e)
        y.append(round(p_sum / 5, 2))
        e += 25
    print(x, y)
    plt.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("Precisão")
    plt.title("Precisão por número de Epochs")
    plt.show()


def main():
    #train()
    graph()

if __name__ == '__main__':
    main()