import tools


def prixMaison(taille):
    return taille * (10 ** 4)


data = ([20, 2], [40, 4], [80, 8], [30, 2.5], [70, 5], [80, 6])
#data += ([150, 6.5], [200, 11], [90, 7.5])
a = (4 - 2) / (40 - 20)
# y=ax+b
b = 2 - a * 20
print('Une maison de 30m² coute ' + str(prixMaison(30)))
print('Une maison de 80m² coute ' + str(prixMaison(80)))
print('Une maison de 90m² coute ' + str(prixMaison(90)))
print(tools.LSE(data, [a, b]))
meilleur = tools.LSE(data, [a, b])
for i in range(-10000, 10000, 1):
    test = a+(i/10000)
    if tools.LSE(data, [test, b]) < meilleur:
        meilleur = tools.LSE(data, [test, b])
        aOpti=test

print(aOpti)
print(tools.LSE(data, [aOpti, b]))
print(tools.reg_lin(data))
tools.plot_data(data, aOpti, b)


