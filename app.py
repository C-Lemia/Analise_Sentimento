import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer

# Carregar o modelo salvo
model = load_model('sentiment_model.h5')

# Carregar o índice de palavras do IMDb para converter textos
word_index = imdb.get_word_index()
tokenizer = Tokenizer(num_words=10000)
tokenizer.word_index = word_index

# Configurações de padding e tamanho da sequência
vocab_size = 10000
max_length = 200
padding_type = 'post'
trunc_type = 'post'

# Função para prever sentimento de um novo texto com mais categorias
def prever_sentimento(novo_texto):
    # Converter o novo texto em sequência de números
    sequencia = tokenizer.texts_to_sequences([novo_texto])
    
    # Padding para garantir o mesmo comprimento das sequências de treino
    sequencia_padded = pad_sequences(sequencia, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    # Fazer a predição com o modelo treinado
    predicao = model.predict(sequencia_padded)[0][0]  # Extrair a probabilidade como um valor escalar
    
    # Verificar onde a predição se encaixa em uma escala de categorias
    if predicao < 0.2:
        return 'Muito Negativo'
    elif 0.2 <= predicao < 0.4:
        return 'Negativo'
    elif 0.4 <= predicao < 0.6:
        return 'Neutro'
    elif 0.6 <= predicao < 0.8:
        return 'Positivo'
    else:
        return 'Muito Positivo'

# Testar com vários textos em português
frases = [
    "Eu estou tão feliz com o resultado final, foi melhor do que esperava!",
    "O dia foi maravilhoso, me diverti muito com meus amigos.",
    "Estou profundamente triste com o que aconteceu, não sei o que fazer.",
    "Fiquei extremamente frustrado com o atraso no projeto.",
    "Tudo foi entregue conforme o combinado, excelente atendimento.",
    "Fiquei surpreso com a qualidade, realmente não esperava algo tão bom.",
    "O produto chegou quebrado e não correspondeu às minhas expectativas.",
    "Foi um dia comum, sem muitas surpresas, mas foi produtivo."
]

for frase in frases:
    resultado = prever_sentimento(frase)
    print(f"Sentimento da frase: \"{frase}\" -> {resultado}")
