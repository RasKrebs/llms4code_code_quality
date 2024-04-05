def split_matrix(matrix):
    # Splitting matrix into 4 quadrants
    size = len(matrix)//2
    upperleft = [[0 for i in range(size)] for j in range(size)]
    upperright = [[0 for i in range(size)] for j in range(size)]
    lowerleft = [[0 for i in range(size)] for j in range(size)]
    lowerright = [[0 for i in range(size)] for j in range(size)]
    for i in range(len(matrix)//2):
        for j in range(len(matrix)//2):
            upperleft[i][j] = matrix[i][j]
            upperright[i][j] = matrix[i][j+size]
            lowerleft[i][j] = matrix[i+size][j]
            lowerright[i][j] = matrix[i+size][j+size]
    return upperleft, upperright, lowerleft, lowerright

def add_matrices(matrix1, matrix2):
    # Adding two matrices element-wise
    result = [[0 for i in range(len(matrix1))] for j in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            result[i][j] = matrix1[i][j] + matrix2[i][j]
    return result

def subtract_matrices(matrix1, matrix2):
    # Subtracting two matrices element-wise
    result = [[0 for i in range(len(matrix1))] for j in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            result[i][j] = matrix1[i][j] - matrix2[i][j]
    return result

def strassen_multiplication(matrix1, matrix2):
    # Base case of recursion when the size of matrices is 1x1
    if len(matrix1) == 1:
        return [[matrix1[0][0] * matrix2[0][0]]]

    # Splitting the input matrices
    a, b, c, d = split_matrix(matrix1)
    e, f, g, h = split_matrix(matrix2)

    # Calculating p values for Strassen multiplication algorithm
    p1 = strassen_multiplication(a, subtract_matrices(f, h))
    p2 = strassen_multiplication(add_matrices(a, b), h)
    p3 = strassen_multiplication(add_matrices(c, d), e)
    p4 = strassen_multiplication(d, subtract_matrices(g, e))
    p5 = strassen_multiplication(add_matrices(a, d), add_matrices(e, h))
    p6 = strassen_multiplication(subtract_matrices(b, d), add_matrices(g, h))
    p7 = strassen_multiplication(subtract_matrices(a, c), add_matrices(e, f))

    # Calculating the final result matrices
    upperleft = subtract_matrices(add_matrices(add_matrices(p5, p4), p2), p6)
    upperright = add_matrices(p1, p2)
    lowerleft = add_matrices(p3, p4)
    lowerright = subtract_matrices(subtract_matrices(add_matrices(p5, p1), p3), p7)

    # Constructing the result matrix from its quadrants
    result = [[0 for i in range(len(matrix1))] for j in range(len(matrix1))]
    for i in range(len(upperleft)):
        for j in range(len(upperleft)):
            result[i][j] = upperleft[i][j]
            result[i][j+len(upperright)] = upperright[i][j]
            result[i+len(lowerright)][j] = lowerleft[i][j]
            result[i+len(lowerright)][j+len(lowerright)] = lowerright[i][j]
    return result