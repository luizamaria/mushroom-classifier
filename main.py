import dearpygui.dearpygui as dpg
from ui import setup_ui


def main():
    dpg.create_context() # inicializa o contexto do DearPyGui
   
    with dpg.window(label="Mushroom Classifier", width=1430, height=1000):
        setup_ui() # configurar todos os elementos da interface gráfica (UI)
       
    dpg.create_viewport(title='Neural Network Training Interface', width=800, height=600) # criando um viewport (a área onde a interface gráfica será renderizada)
    dpg.setup_dearpygui() # configura o DearPyGui para uso
    dpg.show_viewport() # exibe a viewport que foi criada anteriormente
    dpg.start_dearpygui() # inicia o loop principal do DearPyGui, renderiza a interface gráfica e a espera interações do usuário
   
    dpg.destroy_context() # limpa e libera os recursos do DearPyGui ao final da execução


if __name__ == "__main__":
    main()
