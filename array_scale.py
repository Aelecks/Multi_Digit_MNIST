import numpy as np

# Helper functions
MM = lambda x,y: np.matmul(x,y)

def rightInverse(A):
    B = np.linalg.inv(np.matmul(A,A.transpose()))
    A_right = np.matmul(A.transpose(),B)
    return A_right

def arrayScale1D(V1,V2,axis=0):
    # The purpose of this function is to output a Matrix that will be used to scale (1D) an array to a new size
    match axis:
        case 0:
            a1,a2 = V1[0,:], V2[0,:] # Case 0 scales along row vector (increases columns)
        case 1:
            a1,a2 = V1[:,0], V2[:,0] # Case 1 scales along the column vector (increases rows)
        
    s1 = min(a1.size, a2.size)
    s2 = max(a1.size, a2.size)

    A = np.zeros((s1,s2))
    size_factor = s2/s1
    current_cost = size_factor
    
    row,column = 0,0
    new_row = False
    while (row < s1) and (column < s2):
        if new_row:
            A[row,column] = (1-current_cost)
            current_cost = size_factor - (1-current_cost)
            column +=1
            new_row = False
            continue
        if current_cost > 1:
            A[row,column] = 1
            column +=1
            current_cost -=1
            continue
        if current_cost < 1:
            A[row,column] = current_cost
            row +=1
            new_row = True
            continue
        if current_cost == 1:
            A[row,column] = 1
            row +=1
            column +=1
            current_cost = size_factor
            
    if a1.size > a2.size:
        return rightInverse(A)
    return A

def arrayScale2D(A,array_size):
    C = np.zeros(array_size)
    B1 = arrayScale1D(A,C, axis=0)
    B2 = arrayScale1D(A,C, axis=1)

    b = np.matmul(A,B1)
    out = np.matmul(B2.transpose(),b)
    return out
