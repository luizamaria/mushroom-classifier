import dearpygui.dearpygui as dpg
from model import load_data, show_first_rows, split_data,train_neural_network, evaluate

def setup_ui():
    with dpg.child(label="Seção de Carregamento de Dados", width=400, height=200):
        dpg.add_button(label="Carregar Dados", callback=load_and_show_data)

    with dpg.child(label="Seção de Configuração da Rede Neural", width=400, height=200):
        dpg.add_input_text(label="hidden_layer_sizes", hint="Ex: (100,)", source="hidden_layer_sizes")
        dpg.add_input_text(label="activation", hint="Ex: 'relu'", source="activation")
        dpg.add_input_text(label="solver", hint="Ex: 'adam'", source="solver")
        dpg.add_input_text(label="learning_rate_init", hint="Ex: 0.001", source="learning_rate_init")
        dpg.add_input_text(label="max_iter", hint="Ex: 200", source="max_iter")

    with dpg.child(label="Seção de Treinamento e Teste", width=400, height=200):
        dpg.add_input_float(label="test_size", source="test_size")
        dpg.add_button(label="Treinar Rede Neural", callback=train_neural_network_callback)
        dpg.add_text("Logs de Treinamento:")
        dpg.add_text("", id="training_logs", wrap=500)

    with dpg.child(label="Seção de Visualização de Resultados", width=400, height=200):
        dpg.add_text("Acurácia do Modelo:")
        dpg.add_text("", id="accuracy", wrap=500)

def load_and_show_data():
    X, y = load_data()
    show_first_rows(X, y)

def train_neural_network_callback():
    hidden_layer_sizes = eval(dpg.get_value("hidden_layer_sizes"))
    activation = dpg.get_value("activation")
    solver = dpg.get_value("solver")
    learning_rate_init = float(dpg.get_value("learning_rate_init"))
    max_iter = int(dpg.get_value("max_iter"))
    test_size = float(dpg.get_value("test_size"))

    # Aqui você implementaria a divisão de dados usando split_data(X, y, test_size)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    mlp = train_neural_network(X_train, y_train, hidden_layer_sizes, activation, solver, learning_rate_init, max_iter)
    accuracy = evaluate(mlp, X_test, y_test)

    dpg.set_value("training_logs", f"Modelo treinado com acurácia: {accuracy:.2f}")
    dpg.set_value("accuracy", f"Acurácia do Modelo: {accuracy:.2f}")
