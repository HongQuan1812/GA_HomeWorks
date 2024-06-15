import matplotlib.pyplot as plt
import numpy as np
import random


# ## số biến của cá thể: 
# D = [2,10]
# ## kích thước quần thể:
# N = [16,32,64,128,256]
# ## [BL, BU]: miền giá trị của biến

# ## max_evaluations
# max_evaluations = [20_000, 100_000]

# "-----------------------DE----------------------"

# ## F: Hệ số scale (create_mutant_vector)
# F = 0.5
# ## Cr: Xác suất lai ghép 
# Cr = 0.7

# "----------------------CEM----------------------"

# sigma = 0.01
# epsilon = 0.001



def Sphere (x):
    """ 
        Global minimum = [0,...,0] -> f(x) = 0
        Input domain: x_i in range(-5.12, 5.12)
    """
    
    return np.sum(x**2)




def Griewank (x):
    """ 
        Global minimum = [0,...,0] -> f(x) = 0
        Input domain: x_i in range(-600, 600)
    """
    
    A = np.sum(x**2/4000)
    B = np.prod( np.cos( x/np.arange(1, len(x)+1) ) )
    
    return A - B + 1




def Rosenbrock (x):
    """ 
        Global minimum = [1,...,1] -> f(x) = 0
        Input domain: x_i in range(-5, 10)
    """
    
    A = 100 * (x[1:] - x[:-1]**2)**2
    B = (x[:-1] - 1)**2
    
    return np.sum(A+B)



def Michalewicz (x, m = 10):
    """ 
        Global minimum: 
            at d = 2 : x = [2.20, 1.57] -> f(x) = -1.8013
            at d = 10:                  -> f(x) = -9.66015
        
        Input domain: x_i in range(0, pi)
    """
    
    A = np.sin(x)
    B = np.sin( (np.arange(1, len(x)+1) * x**2) / np.pi ) ** (2*m)
    
    return -np.sum(A*B)



def Ackley (x, a = 20, b = 0.2, c = 2*np.pi):
    """ 
        Global minimum = [0,...,0] -> f(x) = 0
        Input domain: x_i in range(-32.768, 32.768)
    """
    
    A = a * np.exp( -b * np.sqrt( (1/len(x)) * np.sum(x**2) ) )
    B = np.exp( (1/len(x)) * np.sum( np.cos(c*x) ) ) 
    
    return -A - B + a + np.exp(1)


def initialize_population (N, D, BL, BU):
    """
        N:        Kích thước quần thể
        D:        Số biến – số chiều của một vector (cá thể)
        [BL, BU]: Miền giá trị của biến
    """
    return np.random.uniform(BL, BU, size = (N,D))


def DE_create_mutant_vector(P,F, verbose = False):
    """
        F:   Hệ số scale in range(0,1)
        P:   Quần thể gốc
        P_v: Quần thể đột biến
    """
    N = P.shape[0]
    D = P.shape[1]
    
    # random [r0, r1, r2] cho N cá thể
    indices = list(range(0,N))
    r = np.array([ random.sample(indices, 3) for _ in range(N)])
    
    
    x_r0 = P[r[:,0]]
    x_r1 = P[r[:,1]]
    x_r2 = P[r[:,2]]
    
    # Tạo ra quần thể đột biến
    P_v = x_r0 + F * (x_r1 - x_r2)
    
    if verbose == True:
        print("r:", r)
        print("P:", P)
        print()
        print("x_r0: ", x_r0)
        print("x_r1: ", x_r1)
        print("x_r2: ", x_r2)
        print()
        print("P_v:", P_v)
        
    return P_v


def DE_cross_over (P, P_v, Cr, verbose = False):
    """
        Cr:  Xác suất lai ghép
        P:   Quần thể gốc
        P_v: Quần thể đột biến
        P_u: Quần thể thử nghiệm
    """
    N = P.shape[0]
    D = P.shape[1]
    
    rand   = np.random.uniform( 0, 1, size = P.shape )
    j_rand = np.random.randint( 0, D, size = N )
    
    mask = np.zeros(P.shape)
    mask[rand <= Cr] = 1
    mask[np.arange(0, N) , j_rand] = 1
    
    P_u = np.zeros(P.shape)
    P_u[mask == 1] = P_v[mask == 1]
    P_u[mask == 0] = P[mask == 0]
    
    if verbose == True:
        print("Cr: ", Cr)
        print("rand: ", rand)
        print("j_rand: ", j_rand)
        print()
        print("P:", P)
        print()
        print("P_v:", P_v)
        print()
        print("P_u:", P_u)
    
    return P_u



def DE_selection (P, f_x, P_u, f_u, verbose = False):
    """
        P:   Quần thể gốc
        P_u: Quần thể thử nghiệm
    """
    
    N = P.shape[0]
    D = P.shape[1]
    
    P_new = np.zeros(P.shape)
    f_new = np.zeros(N)

    P_new[f_u < f_x] = P_u[f_u < f_x]
    f_new[f_u < f_x] = f_u[f_u < f_x]

    P_new[f_u >= f_x] = P[f_u >= f_x]
    f_new[f_u >= f_x] = f_x[f_u >= f_x]

    if verbose == True:
        print(f"f_x: {f_x}")
        print(f"f_u: {f_u}")
        print()
        print("P:", P)
        print()
        print("P_u:", P_u)
        print()
        print("P_new:", P_new)
        print()
        print("f_new:", f_new)
    
    return P_new, f_new



def DE (objective, N, D, 
        BL,BU,
        F, Cr,
        max_evaluations, 
        verbose=False,
        create_gif = False):
    
    P = initialize_population (N, D, BL, BU)
    f_x = np.apply_along_axis(objective, axis= 1, arr = P)
    ith = 0   # Thế hệ thứ 0
    
    num_evaluations = len(f_x)
    
    best_fitness = []
    best_fitness.append([num_evaluations, np.max(f_x)])

    if verbose:
        print("#Gen 0:")
        print(f_x)
    
    frames = []
    if create_gif:
        frames.append(P)
        
    while num_evaluations < max_evaluations:
        P_v = DE_create_mutant_vector(P,F)
        
        P_u = DE_cross_over (P, P_v, Cr)
        f_u = np.apply_along_axis(objective, axis= 1, arr = P_u)

        P, f_x = DE_selection (P, f_x, P_u, f_u)
        
        num_evaluations += len(f_u)
        
        best_fitness.append(np.array([num_evaluations, np.max(f_x)]))

        ith += 1
        if verbose:
            print(f'#Gen {ith}:')
            print(f_x)
            
        if create_gif:
            frames.append(P)
            
    if create_gif:
        return np.array(frames)
            
    return (P, f_x, np.array(best_fitness))



def CEM (objective, N, D,
         Ne,
         BL, BU,
         sigma, epsilon,
         max_evaluations, 
         verbose=False,
         create_gif = False):
    
    # Initialization
    C = sigma * np.eye(D)
    
    term1 = np.log(Ne + 1) - np.log( np.arange(1,Ne+1) )
    term2 = np.sum( np.log(Ne + 1) - np.log( np.arange(1,Ne+1) ) )
    w = term1 / term2
    
    u = np.random.uniform(BL, BU, size = D)
    f_u = objective(u)
    
    num_evaluations = 1
    best_fitness = []
    ith = 0
    
    frames_x = []
    frames_C = [C]
    init = u
    
    while num_evaluations < max_evaluations:
        x = np.random.multivariate_normal(mean=u, cov=C, size = N)
        f_x = np.apply_along_axis(objective, axis= 1, arr = x)


        selected_indices = np.argsort(f_x)[:Ne]
        z = x[selected_indices]
        # f_z = f_x[selected_indices]
        
        temp1 = np.array([w[i]*((z[i] - u).reshape(D,1) @ (z[i] - u).reshape(1,D)) for i in range(Ne)])
        C = np.sum(temp1, axis = 0) + epsilon * np.eye(D)
        
        temp2 = np.array([w[i] * z[i] for i in range(Ne)])
        u_new = np.sum(temp2, axis = 0)
        f_u_new = objective(u_new)
        
        if f_u_new < f_u:
            sigma *= (1 - (f_u_new/(f_u+0.000001)))
        else:
            sigma *= (f_u_new/(f_u + 0.000001))
        
        u = u_new.copy()
        f_u = f_u_new
        
        num_evaluations += (len(f_x) + 1)
        best_fitness.append(np.array([num_evaluations, np.max(f_x)]))

        
        if verbose:
            print(f'#Gen {ith}:')
            print(f_x)
            
        ith += 1

        if create_gif:
            frames_x.append(x)
            frames_C.append(C)
    
    if create_gif:
        return ( init, np.array(frames_x), np.array(frames_C) )

    return (x, f_x, np.array(best_fitness))
