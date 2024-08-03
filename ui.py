import dearpygui.dearpygui as dpg
from model import load_data, show_first_rows, split_data, train_neural_network, evaluate, extract_training_metrics

# Variável global para rastrear o número de logs
log_counter = 1

def setup_ui():
    with dpg.child(label="Seção de Carregamento de Dados", width=1400, height=200):
        dpg.add_text("CARREGAMENTO DE DADOS", tag="data",color=(255, 192, 203, 255))
        dpg.add_spacer()
        dpg.add_button(label="Carregar Dados", callback=load_and_show_data)
        dpg.add_text("", tag="first_rows_text", wrap=2000)

    with dpg.child(label="Seção de Configuração da Rede Neural", width=1400, height=180):
        dpg.add_text("CONFIGURAÇÃO DA REDE NEURAL",color=(255, 192, 203, 255))
        dpg.add_spacer()
        dpg.add_input_text(label="Camadas da Rede Neural", hint="Ex: (100,)", tag="hidden_layer_sizes")
        dpg.add_spacer()
        options_act = ['relu', 'identity', 'logistic', 'tanh']
        dpg.add_combo(label="Função de Ativação", items=options_act, tag="activation", default_value=options_act[0])
        dpg.add_spacer()
        options_otim = ['adam','lbfgs', 'sgd']
        dpg.add_combo(label="Otimizador", items=options_otim, tag="solver", default_value=options_otim[0])
        dpg.add_spacer()
        dpg.add_input_text(label="Taxa de Aprendizado", hint="Ex: 0.001", tag="learning_rate_init")
        dpg.add_spacer()
        dpg.add_input_text(label="Máximo de Iterações", hint="Ex: 200", tag="max_iter")

    with dpg.child_window(label="Seção de Treinamento e Teste", width=1400, height=250):
        dpg.add_text("TREINAMENTO E TESTE",color=(255, 192, 203, 255))
        dpg.add_spacer()
        dpg.add_input_float(label="test_size", tag="test_size", default_value=0.3, min_value=0.01, max_value=0.99)
        dpg.add_button(label="Treinar Rede Neural", callback=train_neural_network_callback)
        dpg.add_spacer()
        #dpg.add_text("Logs de Treinamento:", tag="training_logs")

        # Adiciona a tabela de logs de treinamento
        with dpg.table(label="Logs de Treinamento", tag="training_log_table", height=300, width=1400):
            dpg.add_table_column(label="Índice")
            dpg.add_table_column(label="Acurácia")
            dpg.add_table_column(label="Camadas")
            dpg.add_table_column(label="Ativação")
            dpg.add_table_column(label="Otimização")
            dpg.add_table_column(label="Taxa Aprendizado")
            dpg.add_table_column(label="Máx Iterações")
            dpg.add_table_column(label="Tamanho Teste")
            

    with dpg.child_window(label="Seção de Visualização de Resultados", width=1400, height=700):
        dpg.add_text("ACURÁCIA DO MODELO",color=(255, 192, 203, 255))
        with dpg.group(horizontal=True):  # Cria um grupo horizontal
            # Gráfico de Perda
            with dpg.plot(label="Perda ao longo das Épocas", height=400, width=600):
                dpg.add_plot_axis(dpg.mvXAxis, label="Épocas")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Perda")
                dpg.add_line_series([], [], label="Perda de Treinamento", tag="loss_curve_series", parent=y_axis)

            # Gráfico de Acurácia
            with dpg.plot(label="Acurácia ao longo das Épocas", height=400, width=600):
                dpg.add_plot_axis(dpg.mvXAxis, label="Épocas")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Acurácia")
                dpg.add_line_series([], [], label="Acurácia de Treinamento", tag="train_score_series", parent=y_axis)


def load_and_show_data():
    X, y = load_data()
    label_text = show_first_rows(X, y)
    dpg.set_value("first_rows_text", label_text)


def train_neural_network_callback():
    global log_counter
    try:
        camadas_str = dpg.get_value("hidden_layer_sizes")
        learning_rate_init_str = dpg.get_value("learning_rate_init")
        max_iter_str = dpg.get_value("max_iter")
        test_size = dpg.get_value("test_size")


        camadas = eval(camadas_str)
        taxa = float(learning_rate_init_str)
        iter = int(max_iter_str)
        act = dpg.get_value("activation")
        otim = dpg.get_value("solver")

        # Validação dos parâmetros de entrada
        if not camadas_str:
            raise ValueError("Camadas da rede neural não pode ser vazia.")

        if test_size <= 0 or test_size >= 1:
            dpg.set_value("training_logs", "Erro: O tamanho do teste deve estar entre 0 e 1.")
            return

        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size)
        mlp = train_neural_network(X_train, y_train, camadas, act, otim, taxa, iter)
        accuracy = evaluate(mlp, X_test, y_test)
        epochs, loss_curve, train_score = extract_training_metrics(mlp, X_train, y_train)

        #dpg.set_value("accuracy", f"Acurácia do Modelo: {accuracy:.2f}")-
        # Atualize os gráficos com os dados de treinamento
        dpg.set_value("loss_curve_series", (list(epochs), list(loss_curve)))
        dpg.set_value("train_score_series", (list(epochs), list(train_score)))

        # Adiciona uma nova linha na tabela de logs
        try:
            table = dpg.get_item_alias("training_log_table")
            with dpg.table_row(parent=table):
                dpg.add_text(f"{log_counter}")
                dpg.add_text(f"{accuracy:.2f}")
                dpg.add_text(str(camadas))
                dpg.add_text(act)
                dpg.add_text(otim)
                dpg.add_text(f"{taxa}")
                dpg.add_text(f"{iter}")
                dpg.add_text(f"{test_size:.2f}")
        except Exception as e:
            dpg.set_value("training_logs", f"Erro ao adicionar linha na tabela: {e}")
        
        # Incrementa o contador de logs
        log_counter += 1
        
    except ValueError as e:
        dpg.set_value("training_logs", f"Erro de valor: {e}")
    except Exception as e:
        dpg.set_value("training_logs", f"Erro: {e}")
