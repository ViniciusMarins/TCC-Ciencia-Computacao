#imports interface
from PyQt5 import uic,QtWidgets
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QTimer

#imports funcionalidade
import cv2
import os
import dlib
import keyboard
import time
import ProgramaCompleto as pc
import threading
import pandas as pd

#imports modelos
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Aluno, Presenca
import datetime
from sqlalchemy import func

engine = create_engine('mysql://root:root@localhost/db_tcc')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

app = QtWidgets.QApplication([])
tela_principal = uic.loadUi("tela_principal.ui")
tela_captura = uic.loadUi("tela_captura.ui")
tela_listagem = uic.loadUi("tela_listagem.ui")
tela_treinar_loading = uic.loadUi("tela_treinar_loading.ui")
tela_treinar_concluido = uic.loadUi("tela_treinar_concluido.ui")
tela_warning = uic.loadUi("tela_warning.ui")
tela_warning_2 = uic.loadUi("tela_warning_2.ui")
tela_testar_selecionadas = uic.loadUi("tela_testar_selecionadas.ui")
tela_presencas_lancadas = uic.loadUi("tela_presencas_lancadas.ui")
tela_presencas = uic.loadUi("tela_presencas.ui")
tela_lista_presencas = uic.loadUi("tela_lista_presencas.ui")
tela_warning_desconhecido = uic.loadUi("tela_warning_desconhecido.ui")

telas_icon = [tela_captura,tela_listagem,
tela_treinar_concluido,
tela_warning,
tela_warning_2,
tela_testar_selecionadas,
tela_presencas_lancadas,
tela_presencas,
tela_lista_presencas,
tela_warning_desconhecido]

estado = 0

pc.varrer_base_dados()

def icon_images():
    tela_captura.labelIcon.setPixmap(QPixmap("images/capturar_verde.png"))

    tela_listagem.labelIcon.setPixmap(QPixmap("images/relatorio.png"))
    
    tela_testar_selecionadas.labelIcon.setPixmap(QPixmap("images/trabalhar.png"))
    
    tela_warning_2.labelIcon.setPixmap(QPixmap("images/atencao.png"))
    
    tela_treinar_concluido.labelIcon.setPixmap(QPixmap("images/verificar.png"))
    tela_treinar_loading.labelIcon.setPixmap(QPixmap("images/treino.png"))
    
    tela_warning.labelIcon.setPixmap(QPixmap("images/atencao.png"))

    tela_warning_desconhecido.labelIcon.setPixmap(QPixmap("images/atencao.png"))
    
    tela_principal.labelIconFerramenta.setPixmap(QPixmap("images/ferramenta-de-reparacao.png"))
    tela_principal.labelIconLeft.setPixmap(QPixmap("images/comparecimento.png"))
    tela_principal.labelIconRight.setPixmap(QPixmap("images/neural.png"))
    tela_principal.labelIconLogo.setPixmap(QPixmap("images/logo.png"))

    tela_presencas_lancadas.labelIcon.setPixmap(QPixmap("images/verificar.png"))

    tela_lista_presencas.labelIcon.setPixmap(QPixmap("images/verificar.png"))

    for tela in telas_icon:
        tela.btnVoltar.setIcon(QIcon("images/botao-de-logout-delineado.png"))

def tirar_fotos():
    tela_captura.progressBar.show()
    tela_captura.progressBar.setValue(25)

    #Inicializa captura
    captura = cv2.VideoCapture(0)

    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tela_captura.progressBar.setValue(50)

    hog_face_detector = dlib.get_frontal_face_detector()

    # altura e largura que vou usar na hora do rezise
    height = 300

    # calcula proporção entre a largura e a altura da imagem original(a janela que é criada tem essas proporçoes originais)
    ratio = 640 / 480
    tela_captura.progressBar.setValue(75)

    # calcula largura com proporção preservada
    new_width = int(height * ratio)

    contador = 0
    tela_captura.progressBar.setValue(100)
    tela_captura.progressBar.hide()

    prontuario = getProntuarioAluno()
    nome = getNomeAluno()

    path = 'Base_dados/'+nome+'-'+prontuario

    if not os.path.exists(path):
        os.makedirs(path)


    while True:
        # captura um frame
        ret, frame = captura.read()
        
        # inverter a camera [:,::-1]
        frame = cv2.flip(frame,1)

        cv2.imshow('Captura de imagem', frame)

        if keyboard.is_pressed('e'):
            
            image_resized = cv2.resize(frame,(new_width,height), interpolation=cv2.INTER_LANCZOS4)
            faces = hog_face_detector(image_resized)
            
            for face in faces:
                x1 = face.left()
                x2 = face.right()
                y1 = face.top()
                y2 = face.bottom()
                
                # Recorta a região da face da imagem original
                face_img = image_resized[y1:y2, x1:x2]
                face_img_resized = cv2.resize(face_img,(100,100),interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(path+'/'+nome+'_'+str(contador)+'.jpg', face_img_resized)
            contador+=1
            time.sleep(0.4)
        
        
        key = cv2.waitKey(1)
        if key == 27 or key == 113 or key == 81:  # Tecla 'ESC' ou 'q'
            break

    # libera a captura de vídeo
    captura.release()
    cv2.destroyAllWindows()

enabled_botao_captura_style = tela_captura.botao_captura.styleSheet()
def config_iniciais_tela_captura():
    tela_captura.progressBar.hide()
    tela_captura.botao_captura.setEnabled(False)
    tela_captura.botao_captura.setStyleSheet("background-color: gray; color: white;font:700 10pt 'Verdana';border-radius:10px;")

enabled_botao_pronto_style = tela_testar_selecionadas.botao_pronto.styleSheet()
def config_iniciais_tela_testar_selecionadas():
    tela_testar_selecionadas.botao_pronto.setEnabled(False)
    tela_testar_selecionadas.botao_pronto.setStyleSheet("background-color: gray; color: white;font:700 10pt 'Verdana';border-radius:10px;")

def verificar_estado_treinamento():
    global estado
    
    if estado == 1:
        tela_treinar_loading.hide()
        tela_treinar_concluido.progressBar.setValue(100)
        tela_treinar_concluido.show()
        estado = 0
    
    QTimer.singleShot(1000, verificar_estado_treinamento)

def func_treinar_rede():
    pc.varrer_base_dados()
    def loading():
        global estado

        for num in range(0,110,10):
            time.sleep(0.7)
            tela_treinar_loading.progressBar.setValue(num)

        estado = 1

    tela_treinar_loading.show()
    
    loading_thread = threading.Thread(target=loading)  
    train_thread = threading.Thread(target=pc.treinar_rede_neural)

    train_thread.start()
    loading_thread.start()
     
def getNomeAluno():
    nome = tela_captura.input_name.toPlainText().strip().capitalize()
    return nome

def getProntuarioAluno():
    prontuario = tela_captura.input_prontuario.toPlainText().strip().upper()
    return prontuario

def getCampoPesquisa():
    pesquisa = tela_listagem.input_pesquisa.toPlainText().strip().upper()
    return pesquisa

def update_capture_button_state():
    prontuario = getProntuarioAluno()
    nome = getNomeAluno()

    print(nome)
    print(prontuario)

    if len(prontuario) == 9 and len(nome) > 0:
        tela_captura.botao_captura.setEnabled(True)
        tela_captura.botao_captura.setStyleSheet(enabled_botao_captura_style)
    else:
        tela_captura.botao_captura.setEnabled(False)
        tela_captura.botao_captura.setStyleSheet("background-color: gray; color: white;font:700 10pt 'Verdana';border-radius:10px;")

def update_list_alunos_by_filter():
    pesquisa = getCampoPesquisa()
    print(pesquisa)

    tela_listagem.listWidget.clear()
    
    icon = QIcon("images/persona.png")
    
    for prontuario in pc.data.keys():
        if pesquisa == '' or pesquisa in prontuario:
            item = QListWidgetItem(prontuario)
            item.setIcon(icon)
            tela_listagem.listWidget.addItem(item)
            
def update_ready_button_state():
    caminho = getCaminhoFoto()
    print(caminho)

    if len(caminho) >= 4:
        tela_testar_selecionadas.botao_pronto.setEnabled(True)
        tela_testar_selecionadas.botao_pronto.setStyleSheet(enabled_botao_pronto_style)
    else:
        tela_testar_selecionadas.botao_pronto.setEnabled(False)
        tela_testar_selecionadas.botao_pronto.setStyleSheet("background-color: gray; color: white;font:700 10pt 'Verdana';border-radius:10px;")

def listar_alunos():
    tela_listagem.listWidget.clear()
    pc.varrer_base_dados()

    icon_path = "images/persona.png"
    icon = QIcon(icon_path)
    for item_text in pc.data.keys():
        item = QListWidgetItem(item_text)
        item.setIcon(icon)

        tela_listagem.listWidget.addItem(item)

    tela_listagem.show()

def func_graficos():
    try:
        pc.gerar_graficos(pc.historico) 
    except:
        tela_warning.show()

def func_testar_testes():
    try:
        pc.testar_base_teste()
    except:
        tela_warning.show()

def func_testar_selecionadas():
    try:
        pc.testar_fotos_selecionadas(getCaminhoFoto())
    except:
        tela_warning_2.show()

def func_contabilizar_presencas():
    try:
        alunos_detectados = pc.reconhecimento_Presencas(tela_presencas.progressBar)
        print(f"Alunos detectados = {alunos_detectados}")

        if len(alunos_detectados) == 0:
            print("entrei aqui")
            tela_warning_desconhecido.show()
            return
        
        for prontuario,item in pc.data.items():
            if item['nome'] in alunos_detectados:
                print(prontuario)

                try:
                    existing_aluno = session.query(Aluno).filter_by(prontuario=prontuario).first()

                    presenca_count = session.query(func.count(Presenca.id)).filter_by(aluno=existing_aluno).scalar() + 1

                    presenca = Presenca(existing_aluno.prontuario, data=datetime.date.today(), hora=datetime.datetime.now().strftime("%H:%M:%S"))
                    
                    presenca.id = presenca_count
                    # presenca = Presenca(existing_aluno.prontuario,data=datetime.date.today(), hora=datetime.datetime.now().strftime("%H:%M:%S"))

                    session.add(presenca)
                    session.commit()
                except:
                    print('entrei no criando')
                    create_aluno = Aluno(prontuario,item['nome'])

                    presenca = Presenca(create_aluno.prontuario, data=datetime.date.today(), hora=datetime.datetime.now().strftime("%H:%M:%S"))
                    
                    presenca.id = presenca_count
                    # presenca = Presenca(create_aluno.prontuario,data=datetime.date.today(), hora=datetime.datetime.now().strftime("%H:%M:%S"))
                    
                    session.add(create_aluno)
                    session.add(presenca)
                    session.commit()
        tela_presencas_lancadas.show()
    except :
        if len(alunos_detectados) == 0:
            tela_warning_desconhecido.show()

def getCaminhoFoto():
    caminho = tela_testar_selecionadas.input_caminho.toPlainText().strip()
    return caminho

def func_relatorio_csv():
    query = "SELECT prontuario, data, hora FROM presenca"

    df = pd.read_sql_query(query, engine)

    def extrair_hora(texto):
        partes = str(texto).split(' ')
        return partes[2]

    df['hora'] = df['hora'].apply(extrair_hora)

    df.to_csv('lista_de_presenca.csv', index=False, header=False)
    tela_lista_presencas.show()

tela_testar_selecionadas.input_caminho.textChanged.connect(update_ready_button_state)
tela_testar_selecionadas.botao_pronto.clicked.connect(func_testar_selecionadas)

tela_captura.input_name.textChanged.connect(update_capture_button_state)
tela_captura.input_prontuario.textChanged.connect(update_capture_button_state)

tela_listagem.input_pesquisa.textChanged.connect(update_list_alunos_by_filter)

tela_captura.botao_captura.clicked.connect(tirar_fotos)

tela_presencas.progressBar.hide()
tela_presencas.botao_captura.clicked.connect(func_contabilizar_presencas)

tela_principal.btnCapturar_fotos.clicked.connect(tela_captura.show)
tela_principal.btnListar_alunos.clicked.connect(listar_alunos)
tela_principal.btnTreinar_rede.clicked.connect(func_treinar_rede)
tela_principal.btnGraficos.clicked.connect(func_graficos)
tela_principal.btnTestar_teste.clicked.connect(func_testar_testes)
tela_principal.btnTestar_selecionadas.clicked.connect(tela_testar_selecionadas.show)
tela_principal.btnCp_fotos.clicked.connect(tela_presencas.show)
tela_principal.btnCp_video.clicked.connect(func_relatorio_csv)

icon_images()
verificar_estado_treinamento()
config_iniciais_tela_captura()
config_iniciais_tela_testar_selecionadas()
tela_principal.show()
app.exec()