
import os
from collections import Counter
import numpy as np
import re
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

t = ["bebelusi","calculator", "licenta", "apa", "potabila", "odihna", "mici"]
m = len(t)

folder_path = r"C:\Users\Renata\Desktop\TehniciOptimizare\Dataseturi\dataSet_Lab6"
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]

import re

def preprocesare_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text) 
    return text

documente = []
for fisier in fisiere:
    with open(fisier, "r", encoding = "utf-8") as f:
        continut = f.read()
        continut = preprocesare_text(continut) 
        documente.append(continut)

def construieste_matrice(documente, vocabular):
    matrice = []
    for document in documente:
        contor = Counter(document.split())
        vector = [contor[termen] for termen in vocabular]
        matrice.append(vector)
    return np.array(matrice).T

A = construieste_matrice(documente, t)

def TF_IDF(A):
    TF = A / np.maximum(A.max(axis=0), 1)
    df = np.count_nonzero(A, axis=1)
    idf = np.log10(m / np.maximum(df, 1))
    W = TF * idf[:, np.newaxis]
    return W

W = TF_IDF(A)

def aplica_SVD(W, k):
    svd = TruncatedSVD(n_components=k)
    Wk = svd.fit_transform(W.T).T 
    return Wk, svd

print("Termeni:", t)
print("Matrice termen-document (frecvență):")
print(A)

def cautare_termeni(cuvinte_cheie, lista_termeni):
    vector = [1 if termen in cuvinte_cheie else 0 for termen in lista_termeni]
    return np.array(vector).reshape(-1, 1)
cuvinte_cheie = ["bebelusi", "mici"]
vectorQ = cautare_termeni(cuvinte_cheie, t)

print("\nCuvinte căutate:", cuvinte_cheie)
print("Vectorul Q:")
print(vectorQ)

def cauta_documente_relevante(query, tolerante, Wk, svd, t):
    q = np.array([query.count(term) for term in t])
    qp = q @ svd.components_.T
    
    similaritati = similaritatea_cosinus(qp, Wk)

   
    indice_ordonat = np.argsort(similaritati)[::-1]

    precisii = []
    recalls = []
    for tol in tolerante:
        documente_tol_mare = indice_ordonat[similaritati[indice_ordonat] >= tol]
        print(f"Toleranță >= {tol:.1f}:")
        TP = len(set(documente_tol_mare)) 
        FP = len(set(indice_ordonat) - set(documente_tol_mare))  
        FN = len(set(documente_tol_mare) - set(indice_ordonat))  

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precisii.append(precision)
        recalls.append(recall)

    plt.plot(recalls, precisii, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def similaritatea_cosinus(q, A):
    similaritati = []
    for j in range(A.shape[1]):
        document = A[:, j].reshape(-1, 1)
        produs_scalar = np.dot(q.T, document)

        norma_q = np.linalg.norm(q)
        norma_doc = np.linalg.norm(document)

        if norma_q == 0 or norma_doc == 0:
            similaritati.append(0)
        else:
            sim = produs_scalar / (norma_q * norma_doc)
            similaritati.append(sim)
    return np.array(similaritati)


similaritati = similaritatea_cosinus(vectorQ, A)
print("\nSimilarități cosinus:")
print(similaritati)

print("\n")

print("\Ordinea documentelor după similaritate:")
indice_ordonat = np.argsort(similaritati)[::-1]  
for i in indice_ordonat:
    print(f"Document {i} - scor: {similaritati[i]:.4f}")

print("\n")

print("Stabilim nivelul pentru variabila toleranta:")
tolerante = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
for tol in tolerante:
    documente_tol_mare = indice_ordonat[similaritati[indice_ordonat] >= tol]
    print(f"\nToleranță >= {tol:.1f}:")

    for idx in documente_tol_mare:
        print(f"Document {i} - scor: {similaritati[i][0]:.4f}")
  
cuvinte_cheie = ["bebelusi", "mici"]
vectorQ = cautare_termeni(cuvinte_cheie, t)

Wk, svd = aplica_SVD(W, k=3)

tolerante = [0.9, 0.8, 0.7, 0.6, 0.5]

cauta_documente_relevante(cuvinte_cheie, tolerante, Wk, svd, t)


