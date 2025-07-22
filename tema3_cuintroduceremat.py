import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def aplica_svd_si_vizualizeaza(A):
    U, S, Vt = np.linalg.svd(A)

    print("Matricea A:")
    print(A)
    print("\nU:")
    print(U)
    print("\nSigma (valori singulare):")
    print(S)
    print("\nV^T:")
    print(Vt)
    rang_aproximat = np.sum(S > 1e-10)
    print(f"\nRangul aproximat al matricei A: {rang_aproximat}")

    if rang_aproximat < A.shape[0]:
        print("Matricea este deficitară de rang!")
    else:
        print("Matricea este de rang complet.")


    
    A_rec = U @ np.diag(S) @ Vt
    print("\nReconstrucție A (U * Σ * V^T):")
    print(A_rec)
    print("\nReconstrucția este corectă?", np.allclose(A, A_rec))

    if A.shape == (2, 2):
        t = np.linspace(0, 2 * np.pi, 300)
        x = np.cos(t)
        y = np.sin(t)
        puncte = np.vstack((x, y))
        transformate = A @ puncte

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        
        ax[0].plot(x, y, label='Cercul unitate')
        ax[0].quiver(0, 0, Vt[0, 0], Vt[0, 1], color='r', angles='xy', scale_units='xy', scale=1, width=0.01, label='v1')
        ax[0].quiver(0, 0, Vt[1, 0], Vt[1, 1], color='g', angles='xy', scale_units='xy', scale=1, width=0.01, label='v2')
        ax[0].set_title("Cercul")
        ax[0].axis("equal")
        ax[0].grid(True)
        ax[0].legend()

        
        ax[1].plot(transformate[0, :], transformate[1, :], label='Elipsă')
        for i in range(2):
            u_scaled = U[:, i] * S[i]
            ax[1].quiver(0, 0, u_scaled[0], u_scaled[1], color='r' if i == 0 else 'g', angles='xy', scale_units='xy', scale=1, width=0.01, label=f'u{i+1} * σ{i+1}')
        ax[1].set_title("Elipsa")
        ax[1].axis("equal")
        ax[1].grid(True)
        ax[1].legend()

        plt.suptitle("Transformare geometrică prin SVD (2D)")
        plt.show()

    elif A.shape == (3, 3):
        u = np.linspace(0, 2 * np.pi, 300)
        v = np.linspace(0, np.pi, 300)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        xR, yR, zR = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                vec = np.array([x[i, j], y[i, j], z[i, j]])
                rezultat = A @ vec
                xR[i, j], yR[i, j], zR[i, j] = rezultat

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, z, cmap='coolwarm', alpha=0.8, rstride=1, cstride=1, linewidth=0)
        ax1.set_title("Sfera unitate")
        ax1.set_xlim(-10, 10)
        ax1.set_ylim(-10, 10)
        ax1.set_zlim(-10, 10)

        for i in range(3):
            ax1.quiver(0, 0, 0, Vt[i, 0], Vt[i, 1], Vt[i, 2], color='k', linewidth=2, arrow_length_ratio=0.1)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(xR, yR, zR, cmap='coolwarm', alpha=0.8, rstride=1, cstride=1, linewidth=0)
        ax2.set_title("Elipsoidul după SVD")
        ax2.set_xlim(-10, 10)
        ax2.set_ylim(-10, 10)
        ax2.set_zlim(-10, 10)

        for i in range(3):
            u_scaled = U[:, i] * S[i]
            ax2.quiver(0, 0, 0, u_scaled[0], u_scaled[1], u_scaled[2], color='k', linewidth=2, arrow_length_ratio=0.1)

        plt.suptitle("Transformare geometrică prin SVD (3D)")
        plt.show()

    else:
        messagebox.showinfo("Info", "Vizualizarea geometrică e disponibilă doar pentru matrice 2x2 sau 3x3.")

def introdu_matrice_si_ruleaza():
    try:
        dim = simpledialog.askinteger("Dimensiune", "Introdu dimensiunea n (2 sau 3):", minvalue=2, maxvalue=3)
        if dim is None:
            return

        A = []
        for i in range(dim):
            rand = simpledialog.askstring("Linia {}".format(i+1), f"Introdu elementele liniei {i+1} separate prin spațiu:")
            valori = list(map(float, rand.strip().split()))
            if len(valori) != dim:
                raise ValueError("Număr incorect de elemente în linie.")
            A.append(valori)

        A = np.array(A)
        aplica_svd_si_vizualizeaza(A)

    except Exception as e:
        messagebox.showerror("Eroare", str(e))

root = tk.Tk()
root.title("SVD - Interpretare geometrică")

label = tk.Label(root, text="Analiză SVD cu introducere matrice A", font=("Arial", 13))
label.pack(pady=10)

buton_matrice = tk.Button(root, text="Rulează SVD", command=introdu_matrice_si_ruleaza, font=("Arial", 12))
buton_matrice.pack(pady=10)

buton_iesire = tk.Button(root, text="Închide", command=root.destroy, font=("Arial", 12))
buton_iesire.pack(pady=10)

root.mainloop()
