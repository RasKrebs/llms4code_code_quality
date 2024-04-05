def strassens_matrix_mul(x, y):
    # find dimensions of the input matrix
    m = len(x)
    n = len(y[0])

    if not all(len(row) == m for row in x):
        raise ValueError('All rows must have the same size')

    if not all(len(col) == n for col in y):
        return [[sum(x*y for x, y in zip(row_x, col_y))
                 for col_y in zip(*y)]
                for row_x in x]

    if m == 1 or n == 1:
        result = []
        for i in range(m):
            temp = []
            for j in range(n):
                temp.append(sum([a * b for a,b in zip(x[i], y[j])]))
            result.append(temp)
        return result

    else:
        # split the matrices into quarters.
        a = x[:m//2]
        b = [row[:n//2] for row in x]
        c = [row[n//2:] for row in x]
        d = x[m//2:]

        e = y[:n//2]
        f = [row[:n//2] for row in y]
        g = [row[n//2:] for row in y]
        h = y[n//2:]

        p1 = strassens_matrix_mul(a, f)
        p2 = strassens_matrix_mul(a, h)
        p3 = strassens_matrix_mul(b, g)
        p4 = strassens_matrix_mul(c, e)

        p5 = strassens_matrix_mul([[x + y for x,y in zip(*i)] for i in zip(a, c)], [[x + y for x,y in zip(*i)] for i in zip(e, g)])
        p6 = strassens_matrix_mul([[x + y for x,y in zip(*i)] for i in zip(b, d)], [[x + y for x,y in zip(*i)] for i in zip(f, h)])

        p7 = strassens_matrix_mul([[x - y for x,y in zip(*i)] for i in zip(a, b)], [[x - y for x,y in zip(*i)] for i in zip(e, f)])

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p5 + p1 - p3 - p7

    # combine the results back into a single matrix.
    result = []
    for i in range(m//2):
        temp = []
        for j in range(n//2):
            temp += [c11[i][j], c12[i][j]]
        result += [temp, []]

    for i in range(m//2):
        temp = []
        for j in range(n//2):
            temp += [c21[i][j], c22[i][j]]
        result[i+m//2] = temp

    return result