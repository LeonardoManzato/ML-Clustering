# -*- coding: utf-8 -*-

# Análise de Cluster
# Desenvolvido por: Leonardo Manzato
# Data: 18/08/2024

#%% Importando os pacotes

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Carregando o banco de dados 'international-tourist-trips.csv'

# Objetivo: agrupar os países que mais receberam turistas em 2021
# Analisar os 156 países e grupos com maior potencial de turismo

dados_turismo = pd.read_csv('international-tourist-trips.csv')
## Fonte: https://ourworldindata.org/grapher/international-tourist-trips?tab=table

#%% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(dados_turismo.info())

#%% Transformações iniciais no dataset

# Transformação no dataset para manter apenas informações do ano 2021 de cada país

dados_turismo_2021 = dados_turismo[dados_turismo['Year'] == 2021]

# Exclusão das variáveis que não serão utilizadas na criação de clusters

turismo_cluster = dados_turismo_2021.drop(columns=['Code', 'Year'])

# Renomeando variáveis (colunas) para facilitar a descrição

turismo_cluster = turismo_cluster.rename(columns={'Entity': 'País', 'Inbound arrivals (tourists)': '# Tourists'})

# Criação de um dataframe auxiliar para os métodos Elbow e Silhueta
turismo_cluster_aux = turismo_cluster.drop(columns=['País'])

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,7) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(turismo_cluster_aux)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,7)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.axvline(x = 2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,7) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(turismo_cluster_aux)
    silhueta.append(silhouette_score(turismo_cluster_aux, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 7), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster K-means

# Vamos considerar 2 clusters, considerando os resultados dos métodos Elbow e Silhueta

kmeans_final = KMeans(n_clusters = 2, init = 'random', random_state=100).fit(turismo_cluster_aux)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
turismo_cluster['cluster_kmeans'] = kmeans_clusters
turismo_cluster_aux['cluster_kmeans'] = kmeans_clusters
turismo_cluster['cluster_kmeans'] = turismo_cluster['cluster_kmeans'].astype('category')
turismo_cluster_aux['cluster_kmeans'] = turismo_cluster_aux['cluster_kmeans'].astype('category')
turismo_cluster['Year'] = 2021

#%% Identificação das características estatísticas dos 2 clusters formados

# Agrupando o banco de dados

tourism_group = turismo_cluster.groupby(by=['cluster_kmeans'])

# Estatísticas descritivas por grupo

tab_desc_group = tourism_group.describe().T

# Plotando um scatter para identificação dos clusters
fig = px.scatter(turismo_cluster, x='# Tourists', y='Year', 
                 color='cluster_kmeans', 
                 hover_name='País', 
                 title='Scatter Plot', 
                 labels={'x': '# Tourists', 'y': 'Year'},
                 size='# Tourists', size_max=75)

# Ajuste para exibição do eixo Y
fig.update_layout(
    yaxis=dict(
        range=[2021, 2021],  # Define o intervalo do eixo Y
        tickvals=[2021],     # Define os valores dos ticks no eixo Y
        ticktext=['2021']   # Texto correspondente aos ticks no eixo Y
    )
)

# Exibindo o gráfico no browser
fig.show()

#%% Cluster K-means (versão alternativa de estudo)

# Vamos considerar 3 clusters, apesar dos resultados anteriores identificarem 2 como o ideal

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(turismo_cluster_aux)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
turismo_cluster['cluster_kmeans'] = kmeans_clusters
turismo_cluster_aux['cluster_kmeans'] = kmeans_clusters
turismo_cluster['cluster_kmeans'] = turismo_cluster['cluster_kmeans'].astype('category')
turismo_cluster_aux['cluster_kmeans'] = turismo_cluster_aux['cluster_kmeans'].astype('category')
turismo_cluster['Year'] = 2021

#%% Identificação das características estatísticas dos 3 clusters formados

# Agrupando o banco de dados

tourism_group = turismo_cluster.groupby(by=['cluster_kmeans'])

# Estatísticas descritivas por grupo

tab_desc_group = tourism_group.describe().T

# Plotando um scatter para identificação dos clusters
fig = px.scatter(turismo_cluster, x='# Tourists', y='Year', 
                 color='cluster_kmeans', 
                 hover_name='País', 
                 title='Scatter Plot', 
                 labels={'x': '# Tourists', 'y': 'Year'},
                 size='# Tourists', size_max=75)

# Ajuste para exibição do eixo Y
fig.update_layout(
    yaxis=dict(
        range=[2021, 2021],  # Define o intervalo do eixo Y
        tickvals=[2021],     # Define os valores dos ticks no eixo Y
        ticktext=['2021']   # Texto correspondente aos ticks no eixo Y
    )
)

# Exibindo o gráfico no browser
fig.show()