
#!/usr/bin/python
# coding: utf8
import scipy as sp
import matplotlib.pyplot as plt
def mnkGP(x,y):
              d=2 # степень полинома
              fp, residuals, rank, sv, rcond = sp.polyfit(x, y, d, full=True) # Модель
              f = sp.poly1d(fp) # аппроксимирующая функция
              print('Коэффициент -- a %s  '%round(fp[0],4))
              print('Коэффициент-- b %s  '%round(fp[1],4))
              print('Коэффициент -- c %s  '%round(fp[2],4))
              y1=[fp[0]*x[i]**2+fp[1]*x[i]+fp[2] for i in range(0,len(x))] # значения функции a*x**2+b*x+c
              so=round(sum([abs(y[i]-y1[i]) for i in range(0,len(x))])/(len(x)*sum(y))*100,4) # средняя ошибка
              print('Average quadratic deviation '+str(so))
              fx = sp.linspace(x[0], x[-1] + 1, len(x)) # можно установить вместо len(x) большее число для интерполяции
              plt.plot(x, y, 'o', label='Original data', markersize=10)
              plt.plot(fx, f(fx), linewidth=2)
              plt.grid(True)
              plt.show()

def mnkGP1(x,y):
    d=1 # степень полинома
    fp, residuals, rank, sv, rcond = sp.polyfit(x, y, d, full=True) # Модель
    f = sp.poly1d(fp) # аппроксимирующая функция
    print('Коэффициент -- a %s  '%round(fp[0],4))
    print('Коэффициент-- b %s  '%round(fp[1],4))
    #print('Коэффициент -- c %s  '%round(fp[2],4))
    y1=[fp[0]*x[i]+fp[1] for i in range(0,len(x))] # значения функции a*x+b
    so=round(sum([abs(y[i]-y1[i]) for i in range(0,len(x))])/(len(x)*abs(sum(y)))*100,4) # средняя ошибка
    print('Average quadratic deviation '+str(so))
    fx = sp.linspace(x[0], x[-1] + 1, len(x)) # можно установить вместо len(x) большее число для интерполяции
    plt.plot(x, y, 'o', label='Original data', markersize=2)
    plt.plot(fx, f(fx), linewidth=1)
    plt.grid(True)
    plt.show()
##    return fp[0] , fp[1] , so



x=[6.117588043212891, 7.017233371734619, 8.256306648254395, 9.97303581237793, 12.470462799072266]
y=[-2.2326676845550537, -2.425820827484131, -2.5068836212158203, -2.6200978755950928, -2.6848206520080566]


w = [3.4973623752593994, 4.6714582443237305, 6.203658580780029, 8.466578483581543, 11.273311614990234]
e =  [6.0090742111206055, 6.205989837646484, 6.39721155166626, 6.543717861175537, 6.856972694396973]

mnkGP(x,y)
mnkGP1(x,y)

mnkGP(w,e)
mnkGP1(w,e)