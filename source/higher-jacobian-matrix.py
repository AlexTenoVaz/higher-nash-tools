from math import factorial
from itertools import product


# Sympy
from sympy import (
    symbols, diff, SparseMatrix, sqrt
)

# Set of global variables
x, y, z, t = symbols('x y z t')


def custom_sort(
        vectors
        ):
    return sorted(vectors, key=lambda v: (sum(v),) + v[::-1])


def default_col_vec(
        number_of_vars=3, 
        jac_order=3
        ):
    natural_numbers = list(range(jac_order + 1))
    return custom_sort([
        vec for vec in product(natural_numbers, repeat=number_of_vars)
        if 1 <= sum(vec) <= jac_order
    ])

def default_row_vec(
        number_of_vars=3, 
        jac_order=4
        ):
    natural_numbers = list(range(jac_order + 1))
    return custom_sort([
        vec for vec in product(natural_numbers, repeat=number_of_vars)
        if 0 <= sum(vec) <= jac_order - 1
    ])


def derivative_dictionary(
        f, 
        fac = False, 
        col_vectors = None, 
        variables_list = None,  
        number_of_vars=3, 
        jac_order_local=4
        ):
    
    if variables_list is None:
        # por defecto usamos los símbolos globales x,y,z (o menos si number_of_vars<3)
        base_vars = (x, y, z)
        variables_list = list(base_vars[:number_of_vars])

    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    
    if col_vectors is None:
        col_vectors = default_col_vec()

    derivative_dict = {}
    
    if fac == False:
        for vec in col_vectors:
            deriv = f
            for i, order in enumerate(vec):
                deriv = diff(deriv, variables_list[i], order)
            if deriv != 0:
                derivative_dict[vec] = deriv
    else:
        derivative_dict = {}
        for vec in col_vectors:
            deriv = f
            factorial_vector = 1
            for h in range(len(vec)):
                factorial_vector *= factorial(vec[h])
            for i, order in enumerate(vec):
                deriv = diff(deriv, variables_list[i], order)
            deriv /= factorial_vector
            if deriv != 0:
                derivative_dict[vec] = deriv
    return derivative_dict
    
def jacobian_matrix(
        f, 
        row_vectors = None, 
        col_vectors = None,  
        variables_list=None, 
        number_of_vars=3, 
        jac_order_local=4
        ):
    
    if row_vectors is None:
        row_vectors = default_row_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
        
    if variables_list is None:
        # por defecto usamos los símbolos globales x,y,z (o menos si number_of_vars<3)
        base_vars = (x, y, z)
        variables_list = list(base_vars[:number_of_vars])

    derivative_dict = derivative_dictionary(f, fac=True, col_vectors=col_vectors, variables_list=variables_list,
                                            number_of_vars=number_of_vars, jac_order_local=jac_order_local)
    
    derivative_dict = derivative_dictionary(f, True)
    num_rows = len(row_vectors)
    num_cols = len(col_vectors)
    
    sparse_entries = {}
    
    for r_idx, row in enumerate(row_vectors):
        for c_idx, col in enumerate(col_vectors):
            subtracted_vector = tuple(c - r for r, c in zip(row, col))
            
            if all(entry >= 0 for entry in subtracted_vector):
                matching_derivative = derivative_dict.get(subtracted_vector, 0)
                if matching_derivative != 0:
                    sparse_entries[(r_idx, c_idx)] = matching_derivative

    M = SparseMatrix(num_rows, num_cols, sparse_entries)
    return M