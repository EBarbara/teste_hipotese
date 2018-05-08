
# coding: utf-8

# ## Preparação

# In[98]:


#importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.ion()

# In[2]:


#carregar o dataset
bol_soma = pd.read_csv('data/BOL_SOMA.csv')
especialistas = pd.read_csv('data/Especialistas.csv')


# In[3]:


#gerar o dataset resumido
bol_soma_resumido = bol_soma.copy()
info_cols = [x for x in range(42, 70)]
bol_soma_resumido.drop(bol_soma_resumido.columns[info_cols],axis=1,inplace=True)


# In[4]:


#pré-processar o dataset de gabarito
gabarito = especialistas.sort_values(by='Avaliacao_Media', ascending=False)
gabarito.index = gabarito.index + 1
gabarito.drop(columns=['Proposta'], inplace=True)


# In[5]:


#definir tamanho da resposta
resp_size = 21


# In[6]:


def evaluate(dataset, gabarito):
    hit = 0
    for proposta in n_result_cut.index:
        if(proposta in gabarito_cut.index):
            print(f'Hit on {proposta}')
            hit += 1
    print(f'{hit} hits')


# ## Método N

# In[10]:


n_result = pd.DataFrame(
    [[bol_soma_resumido[f'{x}'].sum()]for x in range(1, 43)], 
    columns=['avaliacao'], 
    index=range(1, 43)).sort_values(by='avaliacao', ascending=False)


# ## Método P

# In[ ]:


bol_std = StandardScaler().fit_transform(bol_soma)


# In[78]:


pca = PCA(n_components=3)
matriz = bol_soma.values[:, 42:70]
matriz_std = StandardScaler().fit_transform(matriz)
pca.fit(matriz_std)
print(sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)


# In[91]:


#Running PCA and reconstructing
bol_pca = pca.transform(matriz_std)
bol_reconstruct = pca.inverse_transform(bol_pca)


# In[111]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(bol_pca)
classes = kmeans.labels_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for index, row in enumerate(bol_pca):
    if classes[index] == 0:
        color = 'b'
    else:
        color = 'r'
    
    ax.scatter(row[0], row[1], row[2], color=color)


# In[102]:


#for i in bol_reconstruct:
#    print(i)
print(bol_pca.shape)


# In[12]:





# In[76]:





# In[77]:


pca.components_

