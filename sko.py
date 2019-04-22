#!/usr/bin/python
# coding: utf8
import scipy as sp
import matplotlib.pyplot as plt
import math
def mnkGP(x,y):
              d=2 # степень полинома
              fp, residuals, rank, sv, rcond = sp.polyfit(x, y, d, full=True) # Модель
              print (fp)
              f = sp.poly1d(fp) # аппроксимирующая функция
              print('Коэффициент -- a %s  '%round(fp[0],4))
              print('Коэффициент-- b %s  '%round(fp[1],4))
              print('Коэффициент -- c %s  '%round(fp[2],4))
              y1 = [ fp[0]*x[i]+fp[1] for i in range(0,len(x))] # значения функции a*x+b
              y2 = [ math.sqrt(abs( fp[0] * x[i]**2 - fp[1]**2 ) ) for i in range (0,len(x))]
              #so1 =  math.sqrt ( sum( [ ( y[i] - y1[i] ) **2  for i in range(0,len(x))])/(len(x)) )
              #print (so1)
              so= round ( sum ( [ abs ( y[i] - y1[i] )  for i in range ( 0 , len(x) ) ] )/(len(x)*abs(sum(y)))*100 , 4) # средняя ошибка
              print('Average quadratic deviation '+str(so))
              fx = sp.linspace(x[0], x[-1] + 1, len(x)) # можно установить вместо len(x) большее число для интерполяции
              plt.plot(x, y, 'o', label='Original data', markersize=2)
              plt.plot(fx, f(fx), linewidth=1)
              plt.grid(True)
              plt.show()
              print (round(5.72485462453154,4))

x=[5.72452974319458, 6.976823329925537, 8.304901123046875, 9.659320831298828,
    12.242971420288086]
y=[-2.348968029022217, -2.1011135578155518, -1.6414023637771606, -1.3300329446792603,
    -1.0001020431518555] # данные для проверки по функции y=1/x
print(x)
mnkGP(x,y)