def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y, x))
        else:
            line.append((x, y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line


def graph_to_picture(n_array, pressure_array, resolution, picture):
    tmp_x, tmp_y = None, None
    for p_i in range(len(pressure_array)):
            x = int(n_array[p_i] * (resolution-1))
            y = int(pressure_array[p_i] * (resolution-1))
            picture[x][y] = 1
            if tmp_x is not None:  # connect points with line
                line = bresenham_line(tmp_x, tmp_y, x, y)
                for a, b in line:
                    picture[a][b] = 1
            tmp_x, tmp_y = x, y
