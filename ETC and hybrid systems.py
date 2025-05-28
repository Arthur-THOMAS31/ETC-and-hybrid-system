import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import solve_sylvester
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec


ETC = True 

"""
Définition des fonctions utiles
"""
En = lambda X, z, v, vx,it : np.linalg.norm(X[:,it])**2 + np.linalg.norm(z[it])**2 + np.linalg.norm(v[:,it]*delta_x)**2 + np.linalg.norm(vx[:,it]*delta_x)**2



def discretisation(u, ux ,delta_x, N_x, r,it):
    """
    Applique la discrétisation spatiale sur la matrice u.

    Parameters:
        u : np.ndarray
            Matrice de la variable discrétisée (dimensions: [N_x, N_t]).
        ux : np.darray
            Matrice de la dérivée par rapport à x de u discrétisée.
        delta_x : float
            Pas spatial.
        N_x : int
            Nombre total de points spatiaux.
        r : float
            Facteur de discrétisation spatiale.

    Returns:
        u : np.ndarray
            Matrice de la variable discrétisée mise à jour.
        ux : np.darray
            Matrice de la dérivée par rapport à x de u discrétisée mise à jour.
    """
    for ix in range(1, N_x - 1):
            # Mise à jour de u selon les différences finies
            u[ix, it + 1] = u[ix, it] + r * (u[ix + 1, it] - 2 * u[ix, it] + u[ix - 1, it])
            ux[ix, it+1] = 1/delta_x*(u[ix,it+1]-u[ix-1,it+1])
        # Condition limite : u[N_x-1, k+1] = u[N_x-2, k+1]
        
    u[N_x-1,it+1]=u[N_x-2,it+1] 
    ux[0, it+1] = 1/delta_x*(u[1,it+1]-u[0,it+1])
    ux[N_x-1, it+1] = 1/delta_x*(u[N_x-1,it+1]-u[N_x-2,it+1])
    return u, ux



def euler(X, z, v, vx, delta_t, delta_x, N_t, N_x, A, B, C, r):
    """
    Détermination de l'état du système différentiel grâce à la méthode d'Euler.
    
    Parameters:
        X : np.ndarray
            Matrice des états (dimensions: [len(A), N_t]).
        u : np.ndarray
            Matrice de la variable discrétisée (dimensions: [N_x, N_t]).
        ux : np.darray
            Matrice de la dérivée par rapport à x de u discrétisée (dimensions: [N_x,N_t]).    
        delta_t : float
            Pas temporel.
        delta_x : float
            Pas spatial.
        N_t : int
            Nombre total de points temporels.
        N_x : int
            Nombre total de points spatiaux.
        A : np.ndarray
            Matrice du système dynamique.
        B : np.ndarray
            Matrice du système dynamique.
        C : np.ndarray
            Matrice de couplage.
        r : float
            Facteur de discrétisation spatiale.
    
    Returns:
        X : np.ndarray
            Matrice des états mise à jour.
        u : np.ndarray
            Matrice de la variable discrétisée mise à jour.
        ux : np.darray
            Matrice de la dérivée de u par rapport à x discrétisée mise à jour.
    """
    for it in range(N_t - 1):
        # Mise à jour des états X par Euler
        X[:, it + 1] = X[:, it] + delta_t * (A @ X[:, it] + B.flatten() * u[N_x - 1, it])
        
        # Calcul de u(1, k+1) à partir de C et X[:, k+1]
        u[0, it + 1] = (C @ X[:, it + 1])[0]
        
        # Mise à jour de u en discrétisant
        u, ux = discretisation(u, ux, delta_x, N_x, r,it)
    
    return X, u, ux


def Euler_ETC(X,z,v,vx,delta_t,delta_x,N_t,N_x,Phi,A,B,C,r):
    """
    Détermination de l'état du système différentiel couplé à du ETC grâce à la méthode d'Euler.
    
    Parameters:
        X : np.ndarray
            Matrice des états (dimensions: [len(A), N_t]).
        u : np.ndarray
            Matrice de la variable discrétisée (dimensions: [N_x, N_t]).
        ux : np.darray
            Matrice de la dérivée par rapport à x de u discrétisée (dimensions: [N_x,N_t]).    
        delta_t : float
            Pas temporel.
        delta_x : float
            Pas spatial.
        N_t : int
            Nombre total de points temporels.
        N_x : int
            Nombre total de points spatiaux.
        A : np.ndarray
            Matrice du système dynamique.
        B : np.ndarray
            Matrice du système dynamique.
        C : np.ndarray
            Matrice de couplage.
        r : float
            Facteur de discrétisation spatiale.
    
    Returns:
        X : np.ndarray
            Matrice des états mise à jour.
        u : np.ndarray
            Matrice de la variable discrétisée mise à jour.
        ux : np.darray
            Matrice de la dérivée de u par rapport à x discrétisée mise à jour.
        uk : np.darray
            Matrice de la commande Cpm u(1,tk) discrétisée mise à jour
        Phi : np.darray
            Matrice de la fonction mesurant la déviation Phi
        k : int
            Nombre de fois où la commande a été mise à jour
    """
    k = 0
    it =0
    while it < dt/delta_t-1:
        Phi [it] = ((C@X[:,it])[0]-z[it])**2/ (eta* En(X,z,v,vx,it))
        if (C@X[:,it]-z[it])**2 < eta * En(X,z,v,vx,it):
            X[:,it+1] = X[:,it] + delta_t * (A @ X[:,it] + B.flatten()*(v[N_x-1, it] +z[it]))
            v ,vx = discretisation (v, vx, delta_x, N_x, r,it)
            v [0, it+1] = 0
            vx [N_x-1, it+1] = 0
            z [it+1] = z [it]
            it += 1
            
        if  (C@X[:,it]-z[it])**2 >= eta * En(X,z,v,vx,it):
            Tk .append(it *delta_t) 
            z[it] = (C@X[:,it])[0]
            v[:,it] = v[:,it] - (C@X[:,it] - z[it])
            k +=1
    Phi[N_t-1] =((C@X[:,N_t-1])[0] - z[N_t-1])**2/(eta*En(X,z,v,vx,N_t-1)) 
    Tk.append(dt)
    return X,v,vx,Phi,k


def LMI (A,B,C,gamma):
    """
    Vérification de la condition LMI du théorème.

    Parameters:
        A : np.ndarray
            Matrice du système dynamique.
        B : np.ndarray
            Matrice du système dynamique.
        C : np.ndarray
            Matrice de couplage
        gamma: float
            Coefficient de l'éuqation de la chaleur.
          
    Returns:
         P: np.darray
            Matrice de la fonctionnelle de Lyapunov.
        alpha: float
            Coefficient de la fonctionnelle de Lyapunov.
        beta: float
            Coefficient de la fonctionnelle de Lyapunov.
        eta : float
            Tuning du mécanisme d'échantillonnage. 
    """
    print("Eigenvalues of (A+BC)", np.linalg.eig(A+B@C)[0])
    for k in np.linalg.eig(A+B@C)[0]:
        if k>=0:
            print("A+BC is non singular")
            sys.exit(0)
    a = 0
    if np.linalg.det(A) !=0:
        print (f"eta_max =", (1 + C @ np.linalg.inv(A)@B)**2/(1 + np.linalg.norm(np.linalg.inv(A)@B)**2))
        b = min( 0.5, (1 + C @ np.linalg.inv(A)@B)**2/(1 + np.linalg.norm(np.linalg.inv(A)@B)**2))
    else:
        b =0.5
    j=0
    eta = (a+b)/2
    while j <= 5 or problem.status != "optimal":
        alpha = cp.Variable()
        beta = cp.Variable()
        delta = cp.Variable()
        sigma = cp.Variable()
        P = cp.Variable(np.shape(A), symmetric=True)
        A_cl = A + B@C
        PB = P @ B
        CB = C @ B
        Psi_11 = P @ A_cl +  A_cl.T @ P + eta*sigma*(np.eye(np.shape(A)[0]) + C.T@C)
        Psi_13 = delta*A_cl.T@C.T - PB - eta*sigma*C.T
        Psi = cp.bmat([
        [  Psi_11    ,              PB              ,                             Psi_13                   ],
        [  PB.T      ,   cp.reshape(2*(eta*sigma -alpha*gamma), (1,1))  ,                delta * CB                  ],
        [ Psi_13.T   ,                        beta*CB.T        ,   cp.reshape(-2*delta*CB + (eta -1)*sigma,(1,1))  ]
        ])
        
        constraints = [Psi <<0 ,alpha >=0.1, beta >=0.1, delta >=0.1, sigma>=0.1, P >> 0, alpha*(1 + 2/np.sqrt(eta)) >= delta]
        objective = cp.Minimize(sigma)  
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose = False)
        print ("problem.stauts =", problem.status)
        print(f"alpha =", alpha.value, "beta =", beta.value, "delta =", delta.value, "sigma =", sigma.value, "P =", P.value, "eta =", eta)
        if problem.status == "optimal":
            a = eta
            eta = (a+b)/2
            
        else:
            b = eta 
            eta = (a+b)/2
        j +=1
    return (alpha.value, beta.value, delta.value, P.value,0.184) 


# Initialisation
ex = 6  # Choisir l'exemple à exécuter 
print(f"exemple {ex}")


# Variables selon l'exemple
if ex == 1:
    # Variables pour le couplage
    A = np.array([[0, 0], [3, 1]])
    B = np.array([[-0.8], [-3]])
    C = np.array([[-1, 3]])
    X0 = np.array([10, -5])
    z0 = -1
    Q = np.array([[3, 0], [0, 5]])
    gamma = 1

elif ex == 2:
    A = np.array([[0, 1], [-2, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    Q = np.array([[3, 0], [0, 5]])
    X0 = -np.linalg.inv(A + B @ C) @ B * 1  # np.linalg.inv pour l'inversion de matrice
    X0 = X0.flatten()
    gamma = 0.001

elif ex == 3:
    b = -1
    A = np.array([[0, 1], [-2, -0.1]])
    B = np.array([[0], [b]])
    C = np.array([[1, 0]])
    X0 = np.array([10, -5])
    Q = np.array([[3, 0], [0, 5]])
    gamma = 0.005

elif ex == 4:
    K = -2
    A = np.array([[0, 1], [-1, 1]])
    B = np.array([[0], [1]])
    C = np.array([[0, K]])
    X0 = np.array([10, -3])
    z0 =1
    Q = np.array([[3, 0], [0, 5]])
    gamma = 20

elif ex == 5:

    K = 1
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-10 - K, 10, 0, 0], [5, -15, 0, -0.25]])
    B = np.array([[0], [0], [1], [0]])
    C = np.array([[K, 0, 0, 0]])
    X0 = np.array([0, 1, -1, 0])
    Q = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
    N = np.array([[2],[3], [5], [11]])
    T = 7
    gamma = 1

elif ex ==6:
    A = np.array([[-3, 0], [0, -2]])
    B = np.array([[0.8], [1]])
    C = np.array([[0, 1]])
    X0 = np.array([-5, 10])
    z0 = 2
    Q = np.array([[3, 0], [0, 5]])
    gamma = 1

elif ex == 7:
    K = 0.1
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-10-K, 10, 0, 0], [5, -15, 0, -0.25]])
    B = np.array([[0], [0], [1], [0]])
    C = np.array([[K, 0, 0, 0]])
    X0 = np.array([10, -2, 1, -3])
    Q = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 5, 0], [0, 0, 0, 3]])
    gamma = 10

elif ex ==8:
    A = np.array([[-1, 0, 0, 0, 0, 0, 0, 0 ],
          [0, -2, 0, 0, 0, 0, 0, 0 ],
          [0, 0, -3, 0, 0, 0, 0, 0 ],
          [0, 0, 0, -0.5, 0, 0, 0, 0 ],
          [0, 0, 0, 0, -7, 0, 0, 0 ],
          [0, 0, 0, 0, 0, -11, 0, 0],
          [0, 0, 0, 0, 0, 0, -13, 0],
          [0, 0, 0, 0, 0, 0, 0, -17]])
    B = np.array([[0],
          [-0.8],
          [0.2],
          [1],
          [0],
          [1.3],
          [0.7],
          [1.1]])
    C = np.array([[0, 0.5, 0.2, 0.3, 1, 0, 1.1, 0.7]])
    Q = np.array([[2, 0, 0, 0, 0, 0, 0, 0],
                  [0, 4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 7, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 6, 0, 0, 0],
                  [0, 0, 0, 0, 0, 5, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1.2]])
    X0 = np.array([-1, 2,-3,4,6,-7,2,5])
    gamma = 1


# Discrétisation spatiale et temporelle
dx = 1  # Distance spatiale
dt = 8  # Distance temporelle
N_x = 20  # Nombre de points dans l'espace
Nix = N_x - 1  # Nombre d'intervalles dans l'espace
delta_x = dx / Nix  # Pas spatial

# Pas temporel

if gamma >= 0.1:
    delta_t = delta_x**2 / (2 * gamma)  # Doit être inférieur à une certaine valeur
    # print('delta_t = CFL')
else:
    delta_t = delta_x**2/(2 * 0.1)
    # print('delta_t = CFL_0.1')
if delta_t > 0.01:
    delta_t = 0.01

print('delta_t =',delta_t)

Nit = dt / delta_t  # Nombre d'intervalles dans le temps
N_t = int(np.floor(Nit) + 1)  # Nombre de points temporels

x = np.linspace(0, dx, N_x)
t = np.linspace(0, dt, N_t)
r = (gamma * delta_t) / delta_x**2

#  Initialisation des différentes valeurs V, norme de u, norme de u_x, tk, alpha, beta, P, u, ux, X, Phi, X_0 et u_0
V = np.zeros(N_t)
norm_v_L2 = np.zeros(N_t)
norm_vx_L2 = np.zeros(N_t)
tk=0


if ETC:
    
    Tk=[]
    Tau=[]
    alpha,beta,delta, P, eta = LMI(A,B,C,gamma)

else:
    alpha = 1
    beta = 1
    P = scipy.linalg.solve_sylvester ((A+B@C).T, A+B@C, -Q) 


v = np.zeros((N_x, N_t))
X = np.zeros((len(A), N_t))
z = np.zeros (N_t)
vx = np.zeros ((N_x,N_t))
Phi = np.zeros(N_t)


p0, p1, omega1 = 5, 3, 4

for nx in range(N_x):
    v[nx, 0] = ( (C @ X0).item()- 2 * p0 * (nx / Nix)+ p0 * ((nx / Nix) ** 2)+ p1 * (1 - np.cos(omega1 * np.pi * (nx / Nix)))) 
    # u [nx, 0] = p1
for nx in range (N_x-1):
    vx[nx,0] = 1/delta_x*(v[nx+1,0]-v[nx,0])
v [0,0] = 0
vx[N_x-1,0]=0
X[:, 0] = X0
z[0] = z0


# Calcul de la solution du sytème

if ETC:
    X,v,vx,Phi,k = Euler_ETC(X,z,v,vx,delta_t,delta_x,N_t,N_x,Phi,A,B,C,r)
    print('k =', k)
else :
    X, u, ux = euler(X, v, vx, delta_t, delta_x, N_t, N_x, A, B, C, r)


# Evaluation des résultats

if ETC:
    Tau.append(Tk[0])
    for k in range(1,len(Tk)):
        Tau.append(Tk[k]-Tk[k-1])
    


for it in range (N_t):
    norm_v_L2[it] = np.linalg.norm(v[:,it]*delta_x)
    norm_vx_L2 [it] = np.linalg.norm(vx[:,it]*delta_x)
    V[it] = X[:,it].T@P@X[:,it] + alpha*norm_v_L2[it]**2  + beta*norm_vx_L2[it]**2 + delta * ((C@X[:,it])[0] -z[it])**2


# Représentation graphique

# Tracé de u

fig = plt.figure(1)

ax1 = fig.add_subplot(111, projection="3d")

T, X_mesh = np.meshgrid(t, x)
c=10
ax1.plot_surface(T, X_mesh, v, cmap="viridis")
ax1.set_xlabel("t", fontsize = 30)
ax1.set_ylabel("x", fontsize = 30)
ax1.set_zlabel("u", fontsize = 30)
ax1.set_xticks([0,np.round(dt/3), np.round(2*dt/3),dt])
ax1.set_yticks([0,0.4,0.8,1])
ax1.set_zticks([-c,0,c])
plt.tick_params(axis='x', labelsize=22)  # Ticks de l'axe X
plt.tick_params(axis='y', labelsize=22)  # Ticks de l'axe Y
plt.tick_params(axis = 'z', labelsize = 22)
ax1.set_xlim([0, dt])
ax1.set_ylim([0, dx])
ax1.set_zlim([-c, c])
ax1.set_title("Évolution de u")
ax1.set_box_aspect([1,1,1])
plt.title(f"ex = {ex}, γ = {gamma}, N_x = {N_x}  ", fontsize=14)


# Tracé des états X

fig = plt.figure (2)

ax1 = fig.add_subplot(222)
for i in range(len(A)):
    ax1.plot(t, X[i, :], label=f"$X_{i+1}$")
ax1.set_xlabel("t", fontsize = 30)
plt.tick_params(axis='x', labelsize=30)  # Ticks de l'axe X
plt.tick_params(axis='y', labelsize=30)  # Ticks de l'axe Y
ax1.set_xlim([0, dt])
ax1.set_ylim([-10, 10])
ax1.legend()
ax1.grid()
plt.legend(fontsize = 30)
plt.tight_layout()


# Tracé de V

ax2 = fig.add_subplot(224)
plt.plot(t,V, label = "V(X(t),u(t))")
plt.xlabel("t", fontsize =30)
plt.yscale("log")
plt.tick_params(axis='x', labelsize=30)  # Ticks de l'axe X
plt.tick_params(axis='y', labelsize=30)  # Ticks de l'axe Y
plt.grid()
plt.legend(fontsize=30)

if ETC:

    #  Tracé des intervalles t_{k+1} - t_k

    ax3 = fig.add_subplot(221)
    plt.bar(Tk,Tau,width = 0.02, color = 'black')
    for k in range(len(Tk)):
        plt.scatter(Tk[k],Tau[k], color = 'black', marker = 'x')
    plt.tick_params(axis='x', labelsize=30)  # Ticks de l'axe X
    plt.tick_params(axis='y', labelsize=30)  # Ticks de l'axe Y
    plt.xlabel("t", fontsize= 30)
    plt.ylabel("Interval $t_k - t_{k-1}$", fontsize =30)
    plt.grid()
    ax1.set_xlim([0, dt])
    
    # Tracé de Phi

    ax4 = fig.add_subplot(223)
    plt.plot(t,Phi, label = "$Phi(t)$")
    plt.tick_params(axis='x', labelsize=30)  # Ticks de l'axe X
    plt.tick_params(axis='y', labelsize=30)  # Ticks de l'axe Y
    plt.grid()
    plt.xlabel("t", fontsize = 30)
    ax2.set_xlim([0, dt])
    plt.legend(fontsize = 30)

    # Tracé de la commande échantillonnée u(1,tk) superposée avec son pendant continu u(1,t) 

#     fig = plt.figure(3)
#     plt.plot(t,v[-1,:], label = 'u(1,t)')
#     plt.plot(t,uk, label='$u_{ech}$(1,t)')
#     plt.tick_params(axis='x', labelsize=30)  # Ticks de l'axe X
#     plt.tick_params(axis='y', labelsize=30)  # Ticks de l'axe Y
#     plt.xlabel("t", fontsize =30)
#     plt.grid()
#     plt.legend(fontsize = 30)



plt.show()
