import dearpygui.dearpygui as dpg
from dearpygui.dearpygui import start_dearpygui, create_context, destroy_context, create_viewport, setup_dearpygui, show_viewport
from ui import setup_ui

def main():
    create_context() # inicializa o contexto do DearPyGui, necessário para criar a interface gráfica
    
    with dpg.window(label="Mushroom Classifier", width=800, height=600):
        setup_ui() # configurar todos os elementos da interface gráfica (UI)
    create_viewport(title='Neural Network Training Interface', width=800, height=600) # criando um viewport (a área onde a interface gráfica será renderizada)
    setup_dearpygui() # configura o DearPyGui para uso
    show_viewport() # exibe a viewport que foi criada anteriormente
    start_dearpygui() # inicia o loop principal do DearPyGui, renderiza a interface gráfica e a espera interações do usuário
    
    destroy_context() # limpa e libera os recursos do DearPyGui ao final da execução

if __name__ == "__main__":
    main()