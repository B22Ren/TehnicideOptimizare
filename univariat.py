import numpy as np
from numpy import linalg as la
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Orange_Telecom.csv')

df.drop(['phone_number', 'area_code', 'state', 'intl_plan', 'voice_mail_plan', 'churned'], axis='columns', inplace=True)
corr = df.corr() 
corr['total_intl_minutes'].sort_values() 

corr = df.corr() 
corr['total_intl_calls'].sort_values()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corelație")
plt.show()

 

def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A)
    A = A.reshape(-1, 1)
    A = np.c_[np.ones(A.shape[0]), A]

    maxes = [] 

    for i in range(A.shape[1]):
        maxes.append(np.max(A[:, i]))
        A[:, i] = A[:, i] / maxes[-1]

    b = np.array(df[target])
    b = b.reshape(-1, 1)

    for i in range(1, 9):
        start_time = time.time()
        if i > 1:
            A = np.c_[A, np.power(A, i)]

        inv = la.pinv(np.dot(A.T, A)) 
        x = np.dot(np.dot(inv, A.T), b) 

        pred = np.dot(A, x)

        mse = np.mean(np.square(pred - b))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - b))
        sse = np.sum(np.square(pred - b))
        r2 = 1 - (sse / np.sum(np.square(b - np.mean(b))))

        print(f"Gradul {i}:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"SSE: {sse}")
        print(f"R2: {r2}")
        print(f"Timpul de executie: {time.time() - start_time}")

       
        plt.scatter(A[:, 1], b, alpha=0.5, s=30, color='red', label='Date')
        plt.plot(A[:, 1], np.dot(A, x),color='blue', linewidth=2, label='Linia de regeresie')
        
        plt.xlabel(predictors)
        plt.ylabel(target)
        plt.legend()
        plt.title(f'Regresie Polinomiala(Grad{i})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

solve('total_day_calls', 'total_intl_calls')
