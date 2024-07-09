from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd

def load_data():
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data as pandas dataframe
    X = mushroom.data.features # Características (dados de entrada)
    y = mushroom.data.targets # Rótulos (dados de saída)
    return X, y

def show_first_rows(X, y):
    dpg.add_text("Primeiras 5 linhas do dataset:")
    with dpg.child(height=200):
        for i in range(5):
            dpg.add_text(f"X[{i}]: {X[i]}")
            dpg.add_text(f"y[{i}]: {y[i]}")
            dpg.add_separator()

def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_neural_network(X_train, y_train, hidden_layer_sizes, activation, solver, learning_rate_init, max_iter):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter
    )
    mlp.fit(X_train, y_train)
    return mlp

def evaluate(mlp, X_test, y_test):
    accuracy = mlp.score(X_test, y_test)
    return accuracy
