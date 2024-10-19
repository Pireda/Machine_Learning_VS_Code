# **IMPORTAR BIBLIOTECA
import pandas as pd
import numpy as np
import matplotlib.pyplot
import yellowbrick


#Importar função GaussianNB da biblioteca.pacote 'sklearn.naive_bayes'
from sklearn.naive_bayes import GaussianNB

#Criar Variável
base_risco_credito = pd.read_csv('/Users/saulo/Desktop/Machine Learning/risco_credito_csv.csv')

#Visualizar Variavel
print(base_risco_credito)

# **PRÉ-PROCESSAMENTO**

#Criar Variável Previsora
x_risco_credito = base_risco_credito.iloc[:, 0:4].values
print(x_risco_credito) #Visualizar as variáveis previsoras
print(type(x_risco_credito)) #Verificar se foi convertido para o formato numpy
print(x_risco_credito.shape) #Visualizar a estrutura da variávies (p.ex -> numero de registros e colunas)

#Criar Variável Classe
y_risco_credito = base_risco_credito.iloc[:,4].values
print(y_risco_credito) #Visualizar as variáveis classe
print(type(y_risco_credito)) #Verificar se foi convertido para o formato numpy
print(y_risco_credito.shape) #Visualizar a estrutura da variávies (p.ex -> numero de registros e colunas)

#Converter atributos str por numérico - LabelEncoder

 # Importar função LabelEncoder da biblioteca.pacote 'sklearn.preprocessing'
from sklearn.preprocessing import LabelEncoder

#Criando Variável LabelEncoder
historia = LabelEncoder()
divida = LabelEncoder()
garantias = LabelEncoder()
renda = LabelEncoder()

#Transformar as variáveis str em numérica
x_risco_credito[:,0] = historia.fit_transform(x_risco_credito[:,0])
x_risco_credito[:,1] = divida.fit_transform(x_risco_credito[:,1])
x_risco_credito[:,2] = garantias.fit_transform(x_risco_credito[:,2])
x_risco_credito[:,3] = renda.fit_transform(x_risco_credito[:,3])

print(x_risco_credito)

#Salvar Variável
import pickle
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)

# **CRIANDO ALGORITMO NAIVE BAYES**

#Aplicando na Base Risco de Credito

#Treinamento do Algoritmo
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(x_risco_credito, y_risco_credito) #Gerar tabela de probabilidade

#Tipos de Classe
print(naive_risco_credito.classes_) #['alto' 'baixo' 'moderado']

#Quantidade de ocorrências por classe
print(naive_risco_credito.class_count_)
'''
alto = 6, baixo = 5, moderado = 3
'''

#Probabilidade a priori percentual por classe
print(naive_risco_credito.class_prior_) 
'''
alto = 6 -> 0.42 ou 42% da base de dados,
 baixo = 5 -> 0.35 ou 35% da base de dados, 
 moderado = 3 -> 0.21 ou 21% da base de dados
'''

#Testando o Algoritmo - Previsão de um novos cliente -> Entradas manuais
previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
print(previsao)

previsao_2 = naive_risco_credito.predict([[1,1,1,1], [2,2,0,2]])
print(previsao_2)



#Aplicando a Base Credito

#Carregando o arquivo credit.pkl
with open('credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_test, y_credit_test = pickle.load(f) #criando base de treinamento e teste

#Visualizar o formato da base de treinamento e teste
print(x_credit_treinamento.shape, y_credit_treinamento.shape, x_credit_test.shape, y_credit_test.shape)

#Treinar o Algoritmo 
naive_credit_data = GaussianNB() #Algoritmo Naive Bayes baseado na curva Gauseana ou distribuição normal
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento) #Criando tabela de probabilidade

#Testando o Algoritmo - Respostas Previstas
previsoes_credit_data = naive_credit_data.predict(x_credit_test)
print(previsoes_credit_data)

#Comparando com as Respostas Reais
print(y_credit_test)

#Importar Função de Contagem de Acertos e Erros
#Importar da biblioteca.pacote sklearn.metrics a função accuracy_score
from sklearn.metrics import accuracy_score

#Fazer contagem de acertos e erros
print(accuracy_score(y_credit_test, previsoes_credit_data)) #Acurária de 0.93 ou 93%

#Gerar matriz de confusão
#Importar da biblioteca sklearn.metric a função confusion_matrix
from sklearn.metrics import confusion_matrix

#Gerando matriz de confusão
print(confusion_matrix(y_credit_test, previsoes_credit_data))

'''
#linha 0 = Clientes Pagam
#linha 1 = Clientes Não Pagam
#Coluna 0 = Clientes que Pagam
#Coluna 1 = Clientes Não Pagam

#                       Clientes Pagam       Clientes Não Pagam
#Clientes Pagam               428                    8
#Clientes Não Pagam            23                   41

#INTERPRETAÇÃO: Dos clientes que pagam 8 foram classificados erroneamente como não pagadores
#e dos clientes que não pagam 23 foram classificados erroneamente como pagadores
'''

#Construindo a Matriz de Confusão usando a biblioteca yellowbrick
from yellowbrick.classifier import ConfusionMatrix

''''
IMPORTANTE: O yellowbrick não funciona dentro do ambiente virtual do macOS. 
Para isso, é necessário instalar o pacote distutils, porém, ele foi removido da versão Python 3.12
O distutils era usado por muitos pacotes para verificar as versões do Python e fazer a configuração do ambiente.
A solução que eu encontrei é instalar o pacote setuptools, que contém as ferramentas que anteriormente estavam no distutils
Para instalar digite no terminal o código: pip install setuptools
Se já estiver instalado, pode ser necessário reinstalá-lo ou atualizá-lo: pip install --upgrade setuptools
'''

# #Constrir variável e instancial objeto
# cm = ConfusionMatrix(naive_credit_data) #Constrindo variável da Matriz de Confusão (cm)
# cm.fit(x_credit_treinamento, y_credit_treinamento) #Criando a tabela de probabilidade 
# print(cm.score(x_credit_test, y_credit_test)) #Gerando uma matriz comparativa entre os banco de teste e treinamento 
# cm.show() #Visualizar o gráfico da matriz

#Gerar relatório de classificação do algoritmo

#Importar a função classification_report do sklearn.metrics
# from sklearn.metrics import classification_report

# print(classification_report(y_credit_test, previsoes_credit_data)) ####????ESTÁ DANDO ERRO NESSA PARTE DO CÓDIGO - CHECAR
'''
Para rodar esse código é necessário deixar a o gráfico da matriz de confusão como comentário
'''
#Interpretando os valores  da tabela
'''
recall -> capacidade do algoritmo de reconhecer as variaveis
precision -> capacdade do algoritmo de identificar as variaveis corretamente

O problema do algoritmo foi a baixa capacidade de reconhecer os clientes que não pagam = 0.64
e consequentemente tem baixa capacidade de identificar os clientes que não pagam = 0.84
'''

#Aplicando na Base Census

#Carregando o arquivo census.pkl
with open('census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f) 

#Visualizar o formato da base de treinamento e teste
print(x_census_treinamento.shape, y_census_treinamento.shape, x_census_teste.shape, y_census_teste.shape) # 

#Treinar o Algoritmo 
naive_census = GaussianNB()
naive_census.fit(x_census_treinamento, y_census_treinamento)

#Testando o Algoritmo - Respostas Previstas
previsoes_census = naive_census.predict(x_census_teste)
print(previsoes_census)

#Comparando com as Respostas Reais
print(y_census_teste)

#Fazer contagem de acertos e erros - Acurária
# print(accuracy_score(y_census_teste, previsoes_census)) #Acurária = 0.48 ou 48%

# #Matriz de Confusão
# cm_census = ConfusionMatrix(naive_census)
# cm_census.fit(x_census_treinamento, y_census_treinamento)
# cm_census.score(x_census_teste, y_census_teste)
# cm_census.show()

from sklearn.metrics import classification_report
print(classification_report(y_census_teste, previsoes_census))

#Interpretando os valores  da tabela
'''
recall -> capacidade do algoritmo de reconhecer as variaveis
precision -> capacdade do algoritmo de identificar as variaveis corretamente

O problema do algoritmo foi a baixa capacidade de reconhecer os clientes que não pagam = 0.64
e consequentemente tem baixa capacidade de identificar os clientes que não pagam = 0.84
'''