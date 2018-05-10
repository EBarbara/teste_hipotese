import numpy as np
import pandas as pd
from math import floor
from scipy.stats import ttest_rel, chisquare
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#resumir o dataset
def dataset_shorten(dataset):
    short_dataset = dataset.copy()
    info_cols = [x for x in range(42, 70)]
    short_dataset.drop(short_dataset.columns[info_cols],axis=1,inplace=True)
    return short_dataset
	
#Metodo N a ser usado
def metodo_n(dataset, resp_size):
    dataset_resumido = dataset_shorten(dataset)
    df = pd.DataFrame(
        [[dataset_resumido[f'{x}'].sum()]for x in range(1, 43)], 
        columns=['avaliacao'], 
        index=range(1, 43))
    n_result = []
    i = 0
    while(len(n_result)<resp_size):
        n_result.extend(df.index[df['avaliacao'] == i].tolist())  
        i -= 1
    return n_result
	
# Método P a ser usado
def metodo_p(dataset, resp_size):
    pca = PCA()
    matriz = dataset.values[:, 42:70]
    matriz_std = StandardScaler().fit_transform(matriz)
    pca.fit(matriz_std)
    bol_pca = pca.transform(matriz_std)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(bol_pca)
    classes = kmeans.labels_
    info_cols = [x for x in range(42, 70)]
    short_dataset = dataset_shorten(dataset)
    short_dataset['classe'] = np.asarray(classes)
    df = short_dataset.groupby('classe').sum().transpose().rename(columns={0:'avaliacao_0', 1:'avaliacao_1'})
    
    p_result = []
    i_0 = 0
    i_1 = 0
    cur = 0
    while(len(p_result)<resp_size):
        if(cur == 0):
            while(len(df.index[df['avaliacao_0'] == i_0].tolist()) == 0):
                i_0 -= 1
            p_result.extend(df.index[df['avaliacao_0'] == i_0].tolist())
            i_0 -= 1
            cur = 1
        else:
            while(len(df.index[df['avaliacao_1'] == i_1].tolist()) == 0):
                i_1 -= 1
            p_result.extend(df.index[df['avaliacao_1'] == i_0].tolist())
            i_1 -= 1
            cur = 0
        p_result = list(set(p_result))
    return list(map(int, p_result))

def teste_t(matrix, confidence, hipotese, equal=True):
    n_sample = []
    p_sample = []
    result = pd.DataFrame(columns=['tamanho', hipotese, 'p-value'])
    
    for size in range(30, 81):
        n_sample = [x[1] for x in precision_matrix if x[0] == size]
        p_sample = [x[2] for x in precision_matrix if x[0] == size]
            
        t_stats, t_p_value = ttest_rel(n_sample, p_sample)

        if(equal == False):
            t_p_value /= 2

        size_result = pd.Series(data=[size, 'Rejeitamos h0' if (t_p_value < confidence) else 'Não rejeitamos h0', t_p_value],
                               index=['tamanho', hipotese, 'p-value'])
        result = result.append(size_result,ignore_index=True)
    return result	

if __name__ == "__main__":
	
	#definir tamanho da resposta
	Aval_max = 3 #Serve pra que? Não usamos
	Taxa_compressao = 0.5
	Numero_projetos = 42
	resp_size = int(floor(Taxa_compressao * Numero_projetos))

	#carregar o main dataset
	bol_soma = pd.read_csv('data/BOL_SOMA.csv')
	
	#carregar e pré-processar o dataset de gabarito
	especialistas = pd.read_csv('data/Especialistas.csv')
	gabarito = especialistas.sort_values(by='Avaliacao_Media', ascending=False)
	gabarito.index = gabarito.index + 1
	gabarito.drop(columns=['Proposta'], inplace=True)
	resp_corretas = []
	avaliacoes = [5.0, 4.75, 4.67, 4.5, 4.33, 4.25,
				  4.0, 3.75, 3.67, 3.5, 3.33, 3.25,
				  3.0, 2.75, 2.67, 2.5, 2.33, 2.25, 
				  2.0, 1.75, 1.67, 1.5, 1.33, 1.25,
				  1.0, 0.75, 0.67, 0.5, 0.33, 0.25, 0.0]
	#Tá horrível, mas o for "padrão" não roda em float, então tivemos que iterar
	for i in avaliacoes:
		if(len(resp_corretas)>=resp_size):
			break
		resp_corretas.extend(especialistas.index[especialistas['Avaliacao_Media'] == i].tolist())

	dataset = bol_soma

	precision_matrix = []
	recall_matrix = []
	f_matrix = []

	for i in range(30, 81):
		for z in range(1, 101):
			sample_dataset = dataset.sample(i)
			
			resp_n = metodo_n(sample_dataset, resp_size)
			precision_n = len([proposta for proposta in resp_n if proposta in resp_corretas])/len(resp_n)
			recall_n = len([proposta for proposta in resp_n if proposta in resp_corretas])/len(resp_corretas)
			f_n = 2* (precision_n * recall_n) / (precision_n + recall_n)

			resp_p = metodo_p(sample_dataset, resp_size)
			precision_p = len([proposta for proposta in resp_p if proposta in resp_corretas])/len(resp_p)
			recall_p = len([proposta for proposta in resp_p if proposta in resp_corretas])/len(resp_corretas)
			f_p = 2* (precision_p * recall_p) / (precision_p + recall_p)
			
			precision_matrix.append([i,precision_n,precision_p]) 
			recall_matrix.append([i, recall_n, recall_p])
			f_matrix.append([i, f_n, f_p])
			
	print(teste_t(precision_matrix, 0.05, 'Precisão de P > Precisão de N', False))
	print(teste_t(precision_matrix, 0.05, 'Precisão de P = Precisão de N', True))
	print(teste_t(recall_matrix, 0.05, 'Recall de P > Recall de N', False))
	print(teste_t(recall_matrix, 0.05, 'Recall de P = Recall de N', True))
	print(teste_t(f_matrix, 0.05, 'F de P > F de N', False))
	print(teste_t(f_matrix, 0.05, 'F de P = F de N', True))
	
	print(teste_chi(precision_matrix, 0.05, 'Precisão de P = Precisão de N'))
	print(teste_chi(precision_matrix, 0.05, 'Recall de P = Recall de N'))
	print(teste_chi(precision_matrix, 0.05, 'F de P = F de N'))