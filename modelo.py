import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Carregar o dataset IMDb do Keras
(vocab_train, y_train), (vocab_test, y_test) = imdb.load_data(num_words=10000)

# Função para converter a sequência de números de volta para palavras
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Configurações de padding e tamanho da sequência
vocab_size = 10000
max_length = 200
padding_type = 'post'
trunc_type = 'post'

# Padding nas sequências
X_train_padded = pad_sequences(vocab_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)
X_test_padded = pad_sequences(vocab_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Construir o modelo LSTM
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=128, validation_data=(X_test_padded, y_test), verbose=2)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(X_test_padded, y_test, verbose=2)
print(f"\nAcurácia do conjunto de teste: {test_acc * 100:.2f}%")

# Salvar o modelo treinado para uso posterior
model.save('sentiment_model.h5')

# Plotar a Acurácia
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Plotar a Perda
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()
