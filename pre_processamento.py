# **INSTALAÇÃO DAS BIBLIOTECAS - CODIGO NO TERMINAL **
'''
pip install plotly --upgrade
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib.pyplot
pip install plotly.express
'''
# **IMPORTAR BIBLIOTECA**
import plotly as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# **CRIAR VARIÁVEIS**
base_credit = pd.read_csv('/Users/saulo/Desktop/Machine Learning/credit.csv')
print(base_credit)

# # **ANÁLISE PRÉVIA DOS DADOS - ESTATÍSTICA DESCRITIVA**
# print(base_credit.describe())

# # **VISUALIZAÇÃO GRÁFICA DOS DADOS**
# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
# grafico.show()

# **TRATAMENTOS DE VALORES INCONSISTENTES - VALORES NEGATIVOS**
print(base_credit.loc[base_credit['age'] < 0]) #Identificar os registros negativos no banco de dados

# **Apagar a Coluna 'Age' inteira - Não Recomendado**
base_credit2 = base_credit.drop('age', axis = 1) #axis = 1 -> colunas
print(base_credit2)

# **Apagar Apenas os Valores Inconsistentes do Atributo Age - Não Recomendado**
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index) #Cod. para apagar as colunas
print(base_credit3)
print(base_credit[base_credit['age'] < 0].index) #Cod. para ID se as variaveis inconsistentes foram apagadas
print(base_credit3.loc[base_credit3['age'] < 0]) #Cod. para ID se as variaveis inconsistentes foram apagadas

# **Substituir os Valores Inconsistentes pela Média**
print(base_credit.mean()) #média para todos os atributos do banco de dados
print(base_credit['age'].mean()) #média para de atributos específicos do banco de dados
print(base_credit['age'] [base_credit['age'] > 0]. mean()) #calcular a média excluindo o valor inconsistente (valor negativo)

base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 #Substituindo os valores negativos pela média
print(base_credit.head(50)) #mostrando os 50 primieros registros do banco de dados
print(base_credit[base_credit['age'] < 0]) #Verificando se ainda existe valores negativos no Banco de Dados
# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
# grafico.show() #Visualização gráfica para verificar se os valores negativos foram corrigidos
print(base_credit.describe()) #Refazendo a estatística descritiva para verificar se os valores mínimo de 'Age' ainda está negativo

# **TRATAMENTO DE VALORES FALTANTES (NA)**
print(base_credit.isnull().sum()) #Identificar a quantidade de valores NA por atributos
print(base_credit.loc[pd.isnull(base_credit['age'])]) #Identificar a posição dos valores NA no banco de dados
print(base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)) #Substituir os valores NA pela média das idades
print(base_credit.loc[base_credit['clientid'].isin([29,31,32])]) #Verificar se a alteração foi feita

# **DIVISÃO DA BASE DE DADOS ENTRE VARIÁVEIS PREVISORAS E CLASSE**

#Criar Variável Previsora (x)
x_credit = base_credit.iloc[:, 1:4].values
print(x_credit)

'''
#Função iloc do pandas permiter chamar linhas e colunas do data.frame
#.iloc[linhas, colunas] -> Sinal de : seleciona todas as linhas do data.frame
#Para selecionar um intervalor de colunas indique o número da primeira coluna : ultima coluna
#No pandas a primeira coluna recebe valor 0
#Apesar da coluna default ser uma variável classe ela precisa ser add no intervalor pois trata-se de uma variável upper bound do Python (não é considerada)
#Função .values converte para o formato nunmpy
'''
print(type(base_credit)) #Verifique que a base_credit está no formato pandas
print(type(x_credit)) #Verificar se variavel x_credit está no formato numpy

#Criar Variável Classe (y)
y_credit = base_credit.iloc[:, 4].values
print(y_credit)
print(type(y_credit))

# **ESCALONAMENTO DOS VALORES**

'''
O escalonamento padroniza os valores para uma mesma escala para que os algoritmos baseado em distâncias 
não deem mais peso aos valores maiores
'''

print(x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min()) #Verificar os valores mínimos de cada variável 
print(x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max()) #Verificar os valores máximos de cada variável 

'''
Note a discrepancia entre os valores minimos e máximos entre as variávies. Isso pode gerar interpretações
equivocadas dos algoritmos, pois ele pode dar mais peso as valores maiores.
Para evitar que o algoritmo cometa essa erro é necessário fazer o escalonamento dos dados.
'''

#Escalonamento por Padronização (Standardisation) Formula: x = x - média(x)/sd(x)

#Importando a função "StandardScaler" da biblioteca.pacote "sklearn.preprocessing"
from sklearn.preprocessing import StandardScaler

'''
MUITO IMPORTANTE: The sklearn PyPI package is deprecated use scikit-learn instead 
'''

scaler_credit = StandardScaler() #Criando variável scaler_credit
x_credit = scaler_credit.fit_transform(x_credit) #Aplicando a padronização a Variável Previsora (x_credit)
print(x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min()) #Verificar os valores mínimos de cada variável 
print(x_credit)


#BASE DE DADOS CENSUS

# **CRIAR VARIÁVEL
base_census = pd.read_csv('/Users/saulo/Desktop/Machine Learning/census.csv')
print(base_census)

# **ANÁLISE PRÉVIA DOS DADOS - ESTATÍSTICA DESCRITIVA**
print(base_census.describe())  

# **VERIFICAR VALORES FALTANTES**
print(base_census.isnull().sum())

# **DIVISÃO DA BASE DE DADOS ENTRE VARIÁVEIS PREVISORAS E CLASSE**

#Criar Variável Previsora (x)
x_census = base_census.iloc[:,0:14].values 
print(x_census)
print(type(x_census))

#Criando Variável Classe (y)
y_census = base_census.iloc[:,14].values
print(y_census)
print(type(y_census))

# **TRATAMENTO DE ATRIBUTOS CATEGORICOS**

#LabelEncoder - Converte atributos str para numérico

#Importar função LabelEncoder da biblioteca/pacote 'sklearn.preprocessing'
from sklearn.preprocessing import LabelEncoder

#Criando Variaveis para cada atributo str do banco de dados
workclass = LabelEncoder()
education = LabelEncoder()
marital = LabelEncoder()
occupation = LabelEncoder()
relationship = LabelEncoder()
race = LabelEncoder()
sex = LabelEncoder()
country = LabelEncoder()

#Convertendo atributos str para numérico
x_census[:,1] = workclass.fit_transform(x_census[:,1])
x_census[:,3] = education.fit_transform(x_census[:,3])
x_census[:,5] = marital.fit_transform(x_census[:,5])
x_census[:,6] = occupation.fit_transform(x_census[:,6])
x_census[:,7] = relationship.fit_transform(x_census[:,7])
x_census[:,8] = race.fit_transform(x_census[:,8])
x_census[:,9] = sex.fit_transform(x_census[:,9])
x_census[:,13] = country.fit_transform(x_census[:,13])

print(x_census[0])
print(x_census)

#OneHotEncoder - Coverte valores numéricos entre 0 e 1
'''
Evita que os algoritmos atribuam maior peso aos maiores valores de classes p.ex. Classe 4 é menor que Classe 15
'''

print(np.unique(base_census['workclass'])) #Lista todos os tipos de registros por atributos
print(len(np.unique(base_census['workclass']))) #Contar a quantidade de registros diferentes por atributos

#Importar função OneHotEncoder da biblioteca/pacote 'sklearn.preprocessing'
from sklearn.preprocessing import OneHotEncoder

#Importar função Columnransformer da bibliote/pacote 'sklearn.compose'
from sklearn.compose import ColumnTransformer

#Criando Variavel 
onehotenconder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder= 'passthrough')

'''
#criar variável -> onehotencoder_census
#criar objeto -> ColumnTransformer 
#add parâmetro transformer como lista -> transformer=[]
#add uma str 'OneHot', e OneHotEnconder() (que foi importado anteriormente)
#passar a lista dos atributos que vão ser transformados (mesmo atributos transformado através do LabelEncoder)
#add remainder = 'passthrough' para não apagar os atributos numérico do data.frame
'''

#Recriar a base de dados
x_census = onehotenconder_census.fit_transform(x_census).toarray()
print(x_census[0])
print(x_census.shape)

# **ESCALONAMENTO DE ATRIBUTOS**

from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)
print(x_census[0])

# **DIVISÃO DO BANCO EM BASES DE TREINAMENTO E TESTE**

#Importar função train_test_split da biblioteca.pacote sklearn.model_selection
from sklearn.model_selection import train_test_split

# Para o Banco de dados Credit
#Dividindo os atributos previsores e classe em bases de treinamento e teste
x_credit_treinamento, x_credit_test, y_credit_treinamento, y_credit_test = train_test_split(x_credit, y_credit, 
                                                                                            test_size = 0.25, random_state = 0)
'''
# test_size = 0.25 = tamanho da base de dados teste = 25% do Banco de Dados
# random_state = mantem sempre os mesmos registros na base de treinamento e teste
'''

print(x_credit_treinamento.shape) #Visulizar base de treinamento para os atributos previsores
'''
Saida o terminal = (1500, 3) onde 
    1500 registros = 75% do banco de dados
    3 = número de colunas
'''

print(y_credit_treinamento.shape) #Visulizar base de treinamento para os atributos classe
'''
Saida do terminal = (1500,) onde
    1500 registros = 75% do banco de dados
    , = número de colonas = 1 coluna
'''

print(x_credit_test.shape, y_credit_test.shape) #Visualizar a base de test para os atributos previsores e classe
'''
Saída do terminal = (500, 3) (500,) onde
    500 registros = 25% do banco de dados
    3 e , = número de colunas (3 -> atributos previsores, 1 -> atributos classe)
'''

#Para o Banco de dados Census
#Dividindo os atributos previsores e classe em bases de treinamento e teste
x_census_treinamento, x_census_test, y_census_treinamento, y_census_test = train_test_split(x_census, y_census, test_size = 0.15, 
                                                                                            random_state = 0)
print(x_census_treinamento.shape, y_census_treinamento.shape) #Visulizar base de treinamento para os atributos previsores
print(x_census_test.shape, y_census_test.shape) #Visualizar a base de test para os atributos previsores e classe

# **SALVAR AS BASES DE DADOS**

#Importar Biblioteca pickle para salvar as variávies em disco
import pickle

with open('credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_treinamento,
                y_credit_treinamento,
                x_credit_test,
                x_credit_test], f)
    
with open('census.pkl', mode = 'wb') as f:
    pickle.dump([x_census_treinamento,
                 y_census_treinamento,
                 x_census_test,
                 y_census_test], f)
    
