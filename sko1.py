#!/usr/bin/python
# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'fantasy'
def mnkGP(x,y): # функция которую можно использзовать в програме
              n=len(x) # количество элементов в списках
              s=sum(y) # сумма значений y
              s1=sum([1/x[i] for i in  range(0,n)]) #  сумма 1/x
              s2=sum([(1/x[i])**2 for i in  range(0,n)]) #  сумма (1/x)**2
              s3=sum([y[i]/x[i]  for i in range(0,n)])  # сумма y/x
              a= round((s*s2-s1*s3)/(n*s2-s1**2),3) # коэфициент а с тремя дробными цифрами
              b=round((n*s3-s1*s)/(n*s2-s1**2),3)# коэфициент b с тремя дробными цифрами
              s4=[a+b/x[i] for i in range(0,n)] # список значений гиперболической функции
              so=round(sum([abs(y[i] -s4[i]) for i in range(0,n)])/(n*sum(y))*100,3)   # средняя ошибка аппроксимации
              plt.title('Аппроксимация гиперболой Y='+str(a)+'+'+str(b)+'/x\n Средняя ошибка--'+str(so)+'%',size=14)
              plt.xlabel('Координата X', size=14)
              plt.ylabel('Координата Y', size=14)
              plt.plot(x, y, color='r', linestyle=' ', marker='o', label='Data(x,y)')
              plt.plot(x, s4, color='g', linewidth=2, label='Data(x,f(x)=a+b/x')
              plt.legend(loc='best')
              plt.grid(True)
              plt.show()
x=[5.72452974319458, 6.976823329925537, 8.304901123046875, 9.659320831298828,
    12.242971420288086]
y=[-2.348968029022217, -2.1011135578155518, -1.6414023637771606, -1.3300329446792603,
    -1.0001020431518555] # данные для проверки по функции y=1/x
mnkGP(x,y)