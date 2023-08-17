import numpy as np

# Create a function that finds the center of mass / midpoint of any grid:
def get_midpoint(A):
    total = A.sum()

    row_mid, col_mid = 0,0
    n_rows, n_cols = len(A[:,0]), len(A[0,:])
    
    for i in range(n_rows):
        row_mid += i*A[i,:].sum()
    for i in range(n_cols):
        col_mid += i*A[:,i].sum()
    
    row_mid = round(row_mid/total)
    col_mid = round(col_mid/total)
    return (row_mid,col_mid)

def apply_midpoint(A):
    B = np.copy(A)
    row,col = get_midpoint(B)
    B[row,col] = B.max()+1
    return B
# Given two number midpoints: create a new matrix that allows them to be combined

def combine_numbers(A1,A2,OFFSET=(0,0),debug=False):
    # We make the assumption that A1 and A2 are both of equal shape, and A1 is an (n x n) matrix
    if A1.shape != A2.shape:
        return ValueError("The shapes of both matrices must be the same")
    
    n = A1.shape[0]
    row_offset, col_offset = OFFSET
    row_mid_1,col_mid_1 = get_midpoint(A1)
    row_mid_2,col_mid_2 = get_midpoint(A2)
    
    row_shift = (row_mid_1 + 1) + row_offset + (n-(row_mid_2+1))-n
    col_shift = (col_mid_1 + 1) + col_offset + (n-(col_mid_2+1))-n
    
    rows = n + abs(row_shift)
    cols = n + abs(col_shift)
    
    
    origin_start = 0
    origin_end = n
    
    if row_shift < 0:
        row_a1_start, row_a1_end = abs(row_shift) + np.array([origin_start,origin_end])
        row_a2_start, row_a2_end = np.array([origin_start,origin_end])
    if row_shift > 0:
        row_a1_start, row_a1_end = np.array([origin_start,origin_end])
        row_a2_start, row_a2_end = abs(row_shift) + np.array([origin_start,origin_end])
    if row_shift == 0:
        row_a1_start, row_a1_end = np.array([origin_start,origin_end])
        row_a2_start, row_a2_end = np.array([origin_start,origin_end])
    
    if col_shift < 0:
        col_a1_start, col_a1_end = abs(col_shift) + np.array([origin_start,origin_end])
        col_a2_start, col_a2_end = np.array([origin_start,origin_end])
    if col_shift > 0:
        col_a1_start, col_a1_end = np.array([origin_start,origin_end])
        col_a2_start, col_a2_end = abs(col_shift) + np.array([origin_start,origin_end])
    if col_shift == 0:
        col_a1_start, col_a1_end = np.array([origin_start,origin_end])
        col_a2_start, col_a2_end = np.array([origin_start,origin_end])
        
    M = np.zeros((rows,cols))
    M[row_a1_start:row_a1_end , col_a1_start:col_a1_end] += A1
    M[row_a2_start:row_a2_end , col_a2_start:col_a2_end] += A2
    
    if debug == True:
        print(f'origin start {origin_start} row shift {row_shift} col shift {col_shift} rows {rows} cols {cols}')

    return M