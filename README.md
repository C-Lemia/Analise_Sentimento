# **Análise de sentimentos com uma rede neural recorrente (LSTM)**

Esse código implementa análise de sentimentos com uma rede neural recorrente (LSTM). Mais especificamente, ele faz classificação de sentimentos com base em textos de entrada, categorizando os sentimentos em diferentes níveis, como "Muito Negativo", "Negativo", "Neutro", "Positivo" e "Muito Positivo".

- Como dados usamos o IMDb Movie Reviews Dataset, que é um conjunto de dados de análises de filmes. Este dataset é disponibilizado diretamente pela biblioteca Keras.
- O conjunto de dados é dividido em treinamento e teste.

#### Intervalos de probabilidades:
- Muito Negativo: Se a probabilidade prevista for menor que 0.2.
- Negativo: Se a probabilidade estiver entre 0.2 e 0.4.
- Neutro: Se a probabilidade estiver entre 0.4 e 0.6.
- Positivo: Se a probabilidade estiver entre 0.6 e 0.8.
- Muito Positivo: Se a probabilidade for maior que 0.8.

  OBS: No contexto de treinamento de redes neurais, o termo "épocas" se refere ao número de vezes que o algoritmo de treinamento passa por todo o conjunto de dados de treinamento. Durante cada época, o modelo realiza um ajuste dos pesos das conexões entre os neurônios com base nos dados e nos erros observados.

  ![image](https://github.com/user-attachments/assets/0fe81093-377e-43ed-88a9-80ce4e3de12d)
![image](https://github.com/user-attachments/assets/04a07271-e5fc-475a-afbc-9c026ea438de)
