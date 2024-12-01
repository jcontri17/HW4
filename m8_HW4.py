#################################################
#####   IMPORTS & FUNCTION DECLARATIONS     #####
#################################################

###-----IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import kron
from scipy.sparse import eye
from scipy.sparse import csr_matrix



###-----PARAMETERS
n = 8   #grid size
L = 10  #length on each side in pos or neg direction



###-----FUNCTION FOR VISUALIZING MATRICES
def visualize_matrix(matrix, title, save_path=None):
    matrix = matrix.astype(float)  # Ensure the matrix is of numerical type
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix != 0, cmap='gray_r', interpolation='nearest')  # Inverted color scheme
    plt.title(title)
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure    
    plt.show()


###-----CONSTRUCT 1D LAPLACIAN OPERATOR
#n = grid size in one direction
#L = total length on one side

def Lap2D(n,L):
    e = np.ones(n)
    offsets = [-1, 0, 1]
    A = spdiags([e, -2*e, e], offsets, n, n)
    A = lil_matrix(A)   #Converts to a format that allows item assignment
    
    #Apply Period Boundary Conditions
    A[0,-1] = 1
    A[-1,0] = 1

    #Construct the 2D Laplacian
    A = kron(eye(n), A) + kron(A, eye(n))
    A = A.toarray()
    dx = L / n
    A = (1/(dx**2)) * A
    
    return A


###-----CONSTRUCT THE ∂_x OPERATOR
def partial_x(n,L):
    e = np.ones(n)
    offsets = [-1, 1]
    B = spdiags([-e, e], offsets, n, n)
    B = lil_matrix(B)

    #Apply Period Boundary Conditions
    B[0,-1] = -1
    B[-1,0] = 1

    #Construct the partial-x matrix
    B = kron(eye(n), B.toarray())
    B = B.toarray()

    dx = L/n
    B = (1/(2*dx)) * B
    
    return B


###-----CONSTRUCT THE ∂_y OPERATOR
def partial_y(n,L):
    e = np.ones(n)
    offsets = [-1, 1]
    C = spdiags([-e, e], offsets, n, n)
    C = lil_matrix(C)

    #Apply Period Boundary Conditions
    C[0,-1] = -1
    C[-1,0] = 1

    #Construct the partial
    C = kron(C.toarray(), eye(n))
    C = C.toarray()
    dy = L/n
    C = (1/(2*dy)) * C
    
    return C




#################################################
#####               MAIN                    #####
#################################################

L=2*L
A = Lap2D(n,L)
B = partial_x(n,L)
C = partial_y(n,L)

print(A)

###-----SHOW VISUAL REPRESENTATION
visualize_matrix(A, '$\delta_x^2 \delta_y^2$', save_path='dx2dy2_m8.png')
visualize_matrix(B, '$\delta_x$', save_path='dx_m8.png')
visualize_matrix(C, '$\delta_y$', save_path='dy_m8.png')



###-----ANSWERING THE QUESTION
A1 = A
A2 = B
A3 = C
np.save('A1.npy', A)
np.save('A2.npy', B)
np.save('A3.npy', C)

