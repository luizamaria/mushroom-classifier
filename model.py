from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np


def load_data():
    # fetch dataset
    mushroom = fetch_ucirepo(id=73)
    # data as pandas dataframe
    X = mushroom.data.features # Características (dados de entrada)
    y = mushroom.data.targets # Rótulos (dados de saída)
    return X, y

def split_data(X, y, test_size):
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42)
    print("split data")
    return X_train, X_test, y_train, y_test

def train_neural_network(X_train, y_train, camadas, act, otim, taxa, iter):
    print("cheguayyy")
    mlp = MLPClassifier(
        hidden_layer_sizes=camadas,
        activation=act,
        solver=otim,
        learning_rate_init=taxa,
        max_iter=iter,
        verbose=True
    )
    mlp.fit(X_train, y_train)
    return mlp

def evaluate(mlp, X_test, y_test):
    accuracy = mlp.score(X_test, y_test)
    return accuracy

def extract_training_metrics(mlp, X_train, y_train):
    loss_curve = mlp.loss_curve_
    train_score = [mlp.score(X_train, y_train) for _ in range(1, len(loss_curve) + 1)]
    epochs = np.arange(1, len(loss_curve) + 1)
    return epochs, loss_curve, train_score

def show_first_rows(X, y):
    # Traduções dos títulos das características e do rótulo
    feature_names_translation = {
        'cap-shape': 'Formato do Chapéu',
        'cap-surface': 'Superfície do Chapéu',
        'cap-color': 'Cor do Chapéu',
        'bruises': 'Machucados',
        'odor': 'Odor',
        'gill-attachment': 'Anexação das Brânquias',
        'gill-spacing': 'Espaçamento das Brânquias',
        'gill-size': 'Tamanho das Brânquias',
        'gill-color': 'Cor das Brânquias',
        'stalk-shape': 'Formato do Talo',
        'stalk-root': 'Raiz do Talo',
        'stalk-surface-above-ring': 'Superfície do Talo Acima do Anel',
        'stalk-surface-below-ring': 'Superfície do Talo Abaixo do Anel',
        'stalk-color-above-ring': 'Cor do Talo Acima do Anel',
        'stalk-color-below-ring': 'Cor do Talo Abaixo do Anel',
        'veil-type': 'Tipo de Velo',
        'veil-color': 'Cor do Velo',
        'ring-number': 'Número de Anéis',
        'ring-type': 'Tipo de Anel',
        'spore-print-color': 'Cor da Impressão de Esporos',
        'population': 'População',
        'habitat': 'Habitat'
    }
    
    # Dicionário com explicações dos valores
    value_meanings = {
        'cap-shape': {
            'b': 'Bell (Cônico)',
            'c': 'Conical (Cônico)',
            'x': 'Convex (Convexo)',
            'f': 'Flat (Plano)',
            'k': 'Knobbed (Com Botão)',
            's': 'Sunken (Afundado)'
        },
        'cap-surface': {
            'f': 'Fibrous (Fibroso)',
            'g': 'Grooves (Ranhuras)',
            'y': 'Scaly (Escamoso)',
            's': 'Smooth (Liso)'
        },
        'cap-color': {
            'n': 'Brown (Marrom)',
            'b': 'Buff (Bege)',
            'c': 'Cinnamon (Canela)',
            'g': 'Gray (Cinza)',
            'r': 'Red (Vermelho)',
            'p': 'Pink (Rosa)',
            'u': 'Purple (Roxo)',
            'e': 'White (Branco)',
            'w': 'Yellow (Amarelo)'
        },
        'bruises': {
            't': 'Torn (Com Machucados)',
            'f': 'No (Sem Machucados)'
        },
        'odor': {
            'a': 'Almond (Amêndoa)',
            'l': 'Anise (Anis)',
            'c': 'Chemical (Químico)',
            'y': 'Foul (Desagradável)',
            'f': 'Musty (Mofo)',
            'm': 'None (Nenhum)',
            'n': 'Pungent (Azedo)',
            'p': 'Spicy (Picante)'
        },
        'gill-attachment': {
            'a': 'Attached (Anexado)',
            'd': 'Descending (Descendente)',
            'f': 'Free (Livre)',
            'n': 'Notched (Entalhado)'
        },
        'gill-spacing': {
            'c': 'Close (Próximo)',
            'w': 'Crowded (Aglomerado)',
            'd': 'Distant (Distante)'
        },
        'gill-size': {
            'b': 'Broad (Largo)',
            'n': 'Narrow (Estreito)'
        },
        'gill-color': {
            'b': 'Black (Preto)',
            'n': 'Brown (Marrom)',
            'h': 'Buff (Bege)',
            'g': 'Gray (Cinza)',
            'r': 'Red (Vermelho)',
            'o': 'Orange (Laranja)',
            'p': 'Pink (Rosa)',
            'u': 'Purple (Roxo)',
            'e': 'White (Branco)',
            'w': 'Yellow (Amarelo)'
        },
        'stalk-shape': {
            'e': 'Enlarging (Aumentando)',
            't': 'Tapering (Afunilado)'
        },
        'stalk-root': {
            'b': 'Bulbous (Bulboso)',
            'c': 'Club (Clava)',
            'u': 'Cup (Copa)',
            'e': 'Elliptical (Elíptico)',
            'r': 'Rooted (Raiz)'
        },
        'stalk-surface-above-ring': {
            'f': 'Fibrous (Fibroso)',
            'k': 'Knotty (Nó)',
            's': 'Smooth (Liso)',
            'y': 'Scaly (Escamoso)'
        },
        'stalk-surface-below-ring': {
            'f': 'Fibrous (Fibroso)',
            'k': 'Knotty (Nó)',
            's': 'Smooth (Liso)',
            'y': 'Scaly (Escamoso)'
        },
        'stalk-color-above-ring': {
            'b': 'Brown (Marrom)',
            'c': 'Cinnamon (Canela)',
            'g': 'Gray (Cinza)',
            'o': 'Orange (Laranja)',
            'p': 'Pink (Rosa)',
            'r': 'Red (Vermelho)',
            'w': 'White (Branco)',
            'y': 'Yellow (Amarelo)'
        },
        'stalk-color-below-ring': {
            'b': 'Brown (Marrom)',
            'c': 'Cinnamon (Canela)',
            'g': 'Gray (Cinza)',
            'o': 'Orange (Laranja)',
            'p': 'Pink (Rosa)',
            'r': 'Red (Vermelho)',
            'w': 'White (Branco)',
            'y': 'Yellow (Amarelo)'
        },
        'veil-type': {
            'p': 'Partial (Parcial)',
            'u': 'Universal (Universal)'
        },
        'veil-color': {
            'n': 'Brown (Marrom)',
            'o': 'Orange (Laranja)',
            'w': 'White (Branco)'
        },
        'ring-number': {
            'n': 'None (Nenhum)',
            'o': 'One (Um)',
            't': 'Two (Dois)'
        },
        'ring-type': {
            'c': 'Cobwebby (Aranha)',
            'e': 'Evanescent (Evanescente)',
            'f': 'Flaring (Alastrante)',
            'l': 'Large (Grande)',
            'n': 'None (Nenhum)',
            'p': 'Pendant (Pendente)',
            's': 'Sheathing (Envolvente)'
        },
        'spore-print-color': {
            'b': 'Black (Preto)',
            'h': 'Buff (Bege)',
            'o': 'Orange (Laranja)',
            'p': 'Pink (Rosa)',
            'r': 'Red (Vermelho)',
            'u': 'Purple (Roxo)',
            'w': 'White (Branco)',
            'y': 'Yellow (Amarelo)'
        },
        'population': {
            'a': 'Abundant (Abundante)',
            'c': 'Clustered (Aglomerado)',
            'n': 'Numerous (Numeroso)',
            's': 'Scattered (Espalhado)',
            'v': 'Several (Vários)'
        },
        'habitat': {
            'g': 'Grasses (Gramíneas)',
            'l': 'Leaves (Folhas)',
            'm': 'Meadow (Prado)',
            'p': 'Paths (Caminhos)',
            'u': 'Urban (Urbano)',
            'w': 'Waste (Desperdício)',
            'd': 'Woods (Bosques)'
        }
    }

    # Selecionar apenas as primeiras 6 características
    selected_columns = X.columns[:10]
    
    # Traduzir os dados e criar um DataFrame para exibir
    translated_data = {}
    for col in selected_columns:
        translated_col_name = feature_names_translation.get(col, col)
        translated_values = X[col].map(lambda x: value_meanings.get(col, {}).get(x, x))
        translated_data[translated_col_name] = translated_values

    df_translated = pd.DataFrame(translated_data)
    df_translated.index += 1  # Ajustar o índice para começar de 1

    # Criar o texto para exibir
    first_rows_X = df_translated.head().to_string(index=False)
    first_rows_y = y.head().map(lambda x: 'Edible (Comestível)' if x == 'e' else 'Poisonous (Venenoso)').to_string(index=False)
    
    return f"Primeiras 5 linhas do dataset:\n{first_rows_X}\n\nRótulos:\n{first_rows_y}"
