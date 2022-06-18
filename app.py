import cv2, os
import numpy as np

# Arquitetura de modelo de genero
GENDER_MODEL = './modelos/deploy_gender.prototxt'
# Modelo pre-treinado
GENDER_PROTO = 'modelos/gender_net.caffemodel'
# Carregando modelo de predição de genero 
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

GENDER_LIST = ["Masculino", "Feminino"]

# Arquitetura de modelo de faces
FACE_PROTO = "modelos/deploy.prototxt.txt"
FACE_MODEL = "modelos/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# Carregando modelo de predição de faces
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


# Busca faces na imagem baseado no modelo de deteccao de rosto
# e retorna uma lista com as cordenadas da face
def detecta_faces(imagem, confidence_threshold=0.4):
    # Passa o blob para o modelo de deteccao de faces
    blob = cv2.dnn.blobFromImage(imagem, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    # Retorna possiveis faces com base no modelo
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([imagem.shape[1], imagem.shape[0],
                         imagem.shape[1], imagem.shape[0]])
            # Pegando as cordenadas da deteccao do rosto
            start_x, start_y, end_x, end_y = box.astype('int')
            # Aumenta o tamanho do frame de cada face
            start_x, start_y, end_x, end_y = start_x - \
                15, start_y - 15, end_x + 15, end_y + 15
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append as cordenadas da face na lista
            faces.append((start_x, start_y, end_x, end_y))
    return faces


# Carrega a imagem
def carrega_imagem(caminho_da_imagem):
    return cv2.imread(caminho_da_imagem)


# Salva frame da face como Jpg baseado na predição de genero
def crop_imagem(genero, frame):
    cv2.imwrite(f"./{genero}/face_{frame.size}.jpg", frame)



# Verifica genero do frame passado
def verifica_genero(imagem):

    faces = detecta_faces(imagem)
    if faces:
        print(f"Número de faces detectadas: {len(faces)}")
        # Desempacota a lista com as cordenadas das faces
        for (start_x, start_y, end_x, end_y) in faces:
            # Indica posição da face na imagem
            face_img = imagem[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
                227, 227), mean=MODEL_MEAN_VALUES,swapRB=False, crop=True)
            # Passa o blob para o modelo de deteccao de genero
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            numero_do_genero = gender_preds[0].argmax()
            # Define o genero com base na predição do modelo, 0 para Masculio e 1 para Feminino
            genero = GENDER_LIST[numero_do_genero]
            crop_imagem(genero, face_img)
    else:
        print("Nenhuma face foi encontrada na imagem!")

# Carrega imagens dentro de uma determinada pasta
# Ex: carregar_varias_imagens('./images') onde ./images é o local onde está as imagens para processamento
def carregar_varias_imagens(path):
    for imagem in os.listdir(path):
        img_path = os.path.join(path, imagem)        
        imagem =  carrega_imagem(img_path)
        verifica_genero(imagem)

# Para usar essa função, basta remover o comentario da linha abaixo
# e comentar as funções carrega_imagem("./imagens/img4.jpg") e verifica_genero(imagem)
#carregar_varias_imagens('./imagens')

imagem =  carrega_imagem("./imagens/img1.jpg")
verifica_genero(imagem)









