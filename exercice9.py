import tools

data = ([20, 2], [40, 4], [80, 8], [30, 2.5], [70, 5], [80, 6])
data += tools.generate_points(100)

for i in range(1, 8, 1):
    print("degré "+str(i)+" -> "+str(tools.MSE(data, tools.poly_numpy(data, i))))

tools.plot_multi_poly(data, [tools.poly_numpy(data, 1), tools.poly_numpy(data, 2), tools.poly_numpy(data, 3),
                             tools.poly_numpy(data, 4), tools.poly_numpy(data, 5),tools.poly_numpy(data, 6)])

