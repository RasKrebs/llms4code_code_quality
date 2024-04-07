def split(matrix):
    row, col = len(matrix), len(matrix[0])
    mid_row, mid_col = row // 2, col // 2

    a = [[matrix[i][j] for j in range(mid_col)] for i in range(mid_row)]
    b = [[matrix[i][j] for j in range(mid_col, col)] for i in range(mid_row)]
    c = [[matrix[i][j] for j in range(mid_col)] for i in range(mid_row, row)]
    d = [[matrix[i][j] for j in range(mid_col, col)] for i in range(mid_row, row)]

    return a, b, c, d

def add(matrix1, matrix2):
    return [[matrix1[i][j] + matrix2[i][j]  for j in range(len(matrix1))] for i in range(len(matrix1))]

def subtract(matrix1, matrix2):
    return [[matrix1[i][j] - matrix2[i][j]  for j in range(len(matrix1))] for i in range(len(matrix1))]

def strassen(matrix1, matrix2):
    if len(matrix1) == 1:
        return [[matrix1[0][0] * matrix2[0][0]]]

    a, b, c, d = split(matrix1)
    e, f, g, h = split(matrix2)

    p1 = strassen(a, subtract(f, h))
    p2 = strassen(add(a, b), h)
    p3 = strassen(add(c, d), e)
    p4 = strassen(d, subtract(g, e))
    p5 = strassen(add(a, d), add(e, h))
    p6 = strassen(subtract(b, d), add(g, h))
    p7 = strassen(subtract(a, c), add(e, f))

    c11 = add(subtract(add(p5, p4), p2), p6)
    c12 = add(p1, p2)
    c21 = add(p3, p4)
    c22 = subtract(subtract(add(p1, p5), p3), p7)

    result = [[0 for j in range(len(c11)*2)] for i in range(len(c11)*2)]
    for i in range(len(c11)):
        for j in range(len(c11)):
            result[i][j]                   = c11[i][j]
            result[i][j+len(c11)]          = c12[i][j]
            result[i+len(c11)][j]          = c21[i][j]
            result[i+len(c11)][j+len(c11)] = c22[i][j]

    return result
