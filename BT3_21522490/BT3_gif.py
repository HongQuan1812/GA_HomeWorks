import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from BT3_function import Sphere, Griewank, Rosenbrock, Ackley, Michalewicz, DE, CEM


objective = [Sphere, Griewank, Rosenbrock, Michalewicz, Ackley]
B = [(-5.12, 5.12), (-600,600), (-5,10), (0, np.pi), (-32.768, 32.768)]
global_min = [(0, 0), (0, 0), (1, 1), (2.20, 1.57), (0, 0)]

D = [2, 10]
N = [16, 32, 64, 128, 256]
max_evaluations = [20_000, 100_000]

"-----------------------DE----------------------"
F = 0.5
Cr = 0.7
"----------------------CEM----------------------"
sigma = [0.01, 27, 0.3, 0.75, 7.24]
epsilon = 0.001
"----------------------f,n,d--------------------"
f = 4  #function
n = 1  #individuals
d = 0  #variables


    
x_range = np.linspace(B[f][0], B[f][1], 700)
y_range = np.linspace(B[f][0], B[f][1], 700)

# Create a grid of points
X, Y = np.meshgrid(x_range, y_range)
A = np.stack((X,Y), axis=2)
Z = np.apply_along_axis(objective[f], axis=2, arr=A)

np.random.seed(21522490+9)
print("DE!!!")
P_DE = DE (objective[f], N = N[n], D = D[d], 
        BL = B[f][0], BU = B[f][1],
        F = F, Cr = Cr,
        max_evaluations = max_evaluations[d], 
        create_gif = True)

print("Done Done Done DE!!!")

np.random.seed(21522490+9)
print("CEM!!!")
init, P_CEM, C_CEM = CEM (objective[f], N = N[n], D = D[d], 
                        Ne = int(N[n]/2),
                        BL = B[f][0], BU = B[f][1],
                        sigma = sigma[f], epsilon = epsilon,
                        max_evaluations = max_evaluations[d], 
                        create_gif = True)

print("Done Done Done CEM!!!")

# Assuming you have your data arrays X, Y, and Z

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))  # Create a figure and axes object
scatter = []

for i, ax in enumerate(axes.flat):
    # Create the contour plot 
    contour = ax.contour(X, Y, Z, levels=10)  # Adjust levels as needed
    plt.colorbar(contour, label='f(x, y)') 

    # plot the global minimum
    ax.scatter(global_min[f][0], global_min[f][1], color='red')

    # Add labels, title, and colorbar
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(np.min(x_range),np.max(x_range))
    ax.set_ylim(np.min(y_range),np.max(y_range))
    ax.grid(True)

    if i == 0:
        ax.set_title(f'DE')
    else:
        ax.set_title(f'CEM')

    scatter.append(ax.scatter([], [], color='b')) # Initially empty


def init_frame():
    scatter[1].set_offsets(init)
    scatter[1].set_color('b')
    return scatter[1],

def update(frame):
    
    if frame < len(P_DE):
        scatter[0].set_offsets(P_DE[frame, :,:2])  # Set x and y coordinates
        
    if frame < len(P_CEM):
        scatter[1].set_offsets(P_CEM[frame, :,:2])
        scatter[1].set_color('b')
        
        
    return scatter[0], scatter[1],


Num_frames = max(len(P_DE), len(P_CEM))
animation = FuncAnimation(fig = fig,
                        func = update,
                        frames = Num_frames,
                        init_func = init_frame,
                        interval = 100)

plt.show()

