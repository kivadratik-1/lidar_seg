import numpy as np
from numpy import array, abs as np_abs
from open3d import *
from numpy.fft import rfft, hfft, ihfft
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
#from svd_solve import svd, svd_solve
from ransac import *
import os
import matplotlib
import time

import scipy as sp
import matplotlib.pyplot as plt

start_time = time.time()

raw_ang_tab = [ -25 , -1 , -1.667 , -15.639 , -11.31 , 0 , -.667 , -8.843 , -7.254 , 0.333 , -0.333 , -6.148, -5.333, 1.333, 0.667, -4, -4.667, 1.667, 1, -3.667, -3.333, 3.333, 2.333, -2.667, -3, 7, 4.667, -2.333, -2, 15, 10.333, -1.333 ]
angle_tab = sorted(raw_ang_tab)

def find_n_circle(point_cloud):
    ## Функция разбивает поинт клауд на 16 колец (n_beams), возвращает точки с id кольца
    startOri = -1 * math.atan2(point_cloud.points[0][1], point_cloud.points[0][0])
    endOri = -1 * math.atan2(point_cloud.points[(len(point_cloud.points) - 1)][1], point_cloud.points[(len(point_cloud.points) - 1)][0]) + 2 * math.pi;
    count = len(point_cloud.points)
    separated_point_cloud = []
    scans_ids = []
    n_beams = 32
    if (endOri - startOri > 3 * math.pi):
        endOri -= 2 * math.pi
    elif endOri - startOri < math.pi:
        endOri += 2 * math.pi
    #print(startOri , endOri )
    for one_point in point_cloud.points:
        #print(one_point)
        angle = math.atan(one_point[2] / math.sqrt(one_point[0] * one_point[0] + one_point[1] * one_point[1])) * 180 / math.pi
##        print(angle)

        scanID = 0
        #scanID = int((angle + 25) / 2 + 0.5)
        ih = 0
        #print(zu)

        while True:
            ih += 1
            zu = abs(angle_tab[ih - 1]) - abs(angle)
            #print(zu)
            if -0.1 < zu < 0.1:
                scanID = ih - 1
                break

        #print (scanID)
        scans_ids.append(scanID)
        separated_point_cloud.append([one_point, scanID])
    return separated_point_cloud

def find_n_circle_1(point_cloud):
    ## Функция разбивает поинт клауд на 16 колец (n_beams), возвращает точки с id кольца

    startOri = -1 * math.atan2(point_cloud.points[0][1], point_cloud.points[0][0])
    endOri = -1 * math.atan2(point_cloud.points[(len(point_cloud.points) - 1)][1], point_cloud.points[(len(point_cloud.points) - 1)][0]) + 2 * math.pi;
    count = len(point_cloud.points)
    separated_point_cloud = []
    scans_ids = []
    point_oris = []
    n_beams = 321
    if (endOri - startOri > 3 * math.pi):
        endOri -= 2 * math.pi
    elif endOri - startOri < math.pi:
        endOri += 2 * math.pi
    #print(startOri , endOri )
    for one_point in point_cloud.points:
        #print(one_point)
        angle = math.atan(one_point[2] / math.sqrt(one_point[0] * one_point[0] + one_point[1] * one_point[1])) * 180 / math.pi
        point_Ori = math.degrees( math.atan2(one_point[1], one_point[0]))
        scanID = 0
        ih = 0

        while True:
            ih += 1
            zu = abs(angle_tab[ih - 1]) - abs(angle)
            if -0.1 < zu < 0.1:
                scanID = ih - 1
                break

        #print (scanID)
        scans_ids.append(scanID)
        point_oris.append(point_Ori)
        separated_point_cloud.append([one_point, scanID, point_Ori])
    return separated_point_cloud, set(scans_ids), set(point_oris)

def angle_of_point_by_ground(point_from_extended_point_cloud):
    a = [0 , 0 , -1.83]
    #b = [0 , 6.54 , -1.83 ]
    b = point_from_extended_point_cloud
    #d = [0, 6.54, -1.83]
    d =  [point_from_extended_point_cloud[0], point_from_extended_point_cloud[1], -1.83]

    # Координаты точек
    a0 = np.array(a)
    a_inf = np.array(d)
    b_isk = np.array(b)


    a0_a_inf = a_inf - a0
    a0_b_isk = b_isk - a0

    j = np.sum(a0_a_inf *  a0_b_isk)
    m1 = la.norm(a0_a_inf)
    m2 = la.norm(a0_b_isk)


    cos = j / (m1 * m2)
    angle = math.degrees(math.acos(cos))
    #print ('angle =', angle)

    return angle

def ori(point):
    point_Ori = math.degrees( -1 * math.atan2(point[1], point[0]))
    return point_Ori

def massive_of_separated_ring_points(point_cloud_labeled_points, N_ring):
    ## Функция фитьрует точки поинтклауда и возвращает только определенное кольцо
    ring_points = []
    for mas in point_cloud_labeled_points:
        if mas[1] == N_ring:
            ring_points.append(mas[0])
    return ring_points

def find_hyst(array):
    array_of_radiuses = []
    for r in array:
        array_of_radiuses.append(abs(math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])))
    return array_of_radiuses

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    #print(axyz)
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def project_points(x, y, z, a, b, c):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    vector_norm = a*a + b*b + c*c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points = np.column_stack((x, y, z))
    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane


##for file in os.listdir('D:/Dropbox/Inno/Inno/roadedges/'):
##    print (file)

def cutoff_by_z(point_cloud_points, Height_trueshold):
    cuted_point_cloud_points = []
    array_of_radiuses = []
    i=0
    while i < len(point_cloud_points):
        if (point_cloud_points[i][2] < Height_trueshold) and (point_cloud_points[i][0] >= 0) :
            cuted_point_cloud_points.append(point_cloud_points[i])
            #array_of_radiuses.append(point_cloud_points[i][0] * point_cloud_points[i][0] + point_cloud_points[i][1] * point_cloud_points[i][1] + point_cloud_points[i][2] * point_cloud_points[i][2])
            i += 1
        else:
            i += 1
    return cuted_point_cloud_points#, array_of_radiuses

def filtering_by_rings(ring_points, n_ring):
    it = 0
    ring_points_1 = []
    while it+1 < len(ring_points):
       # angle_of_point_by_zero = math.atan(one_point[2] / math.sqrt(one_point[0] * one_point[0] + one_point[1] * one_point[1])) * 180 / math.pi

        first_proizv_dz_po_dy = math.sqrt((ring_points[it+1][2]-ring_points[it][2])**2) / (math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) + 0.00000001)
        first_proizv_dx_po_dy = math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) / (math.sqrt((ring_points[it+1][1]-ring_points[it][1])**2) + 0.00000001)
        ra = math.sqrt((ring_points[it][0])**2+(ring_points[it][1])**2+(ring_points[it][2] -1.83 )**2)
        it +=1
        if n_ring == 0 :
            #if (0.1 < ra < 1.) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 1 :
            #if (0.2 < ra < 9.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 2 :
            #if (0.8 < ra < 120.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 3 :
            #if (0.8 < ra < 122.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 4 :
            #if (0.5 < ra < 142.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 5 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 6 :
           # if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 7 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 8 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 9 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 10 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 11 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 12 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 13 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 14 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 15 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 16 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 17 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        else:
            pass
    #print (len(ring_points_1))
    return ring_points_1

def filtering_by_rings_by_angle(ring_points, n_ring):
    it = 0
    ring_points_1 = []
    while it+1 < len(ring_points):
        first_proizv_dz_po_dy = math.sqrt((ring_points[it+1][2]-ring_points[it][2])**2) / (math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) + 0.00000001)
        first_proizv_dx_po_dy = math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) / (math.sqrt((ring_points[it+1][1]-ring_points[it][1])**2) + 0.00000001)
        ra = math.sqrt((ring_points[it][0])**2+(ring_points[it][1])**2+(ring_points[it][2] -1.83 )**2)
        it +=1
        if n_ring == 0 :
            #if (0.1 < ra < 1.) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 1 :
            #if (0.2 < ra < 9.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 2 :
            #if (0.8 < ra < 120.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 3 :
            #if (0.8 < ra < 122.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 4 :
            #if (0.5 < ra < 142.5) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 5 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 6 :
           # if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 7 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 8 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 9 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 10 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 11 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 12 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 13 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 14 :
            #if (1 < ra < 212) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 15 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 16 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 17 :
            #if (1 < ra < 202) : # and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        else:
            pass
    #print (len(ring_points_1))
    return ring_points_1

def mnkGP(x,y):
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
##    plt.plot(x, y, 'o', label='Original data', markersize=2)
##    plt.plot(fx, f(fx), linewidth=1)
##    plt.grid(True)
##    plt.show()
    return fp[0] , fp[1] , so



################################################################################


if __name__ == '__main__':

    file = '32-3.pcd'

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud('roadedges/'+file)
    #pcd = voxel_down_sample(pcd, voxel_size = 0.05)
    #draw_geometries([pcd])



    ## Part of Roma's code to detect rings:
    ## 1) Разбираем облако точек на 16 лучей

    n_beams_filtered_cloud = PointCloud()
    filtered = []
    #for i in range(3,3):
    filtered = filtered + filtering_by_rings(
                            cutoff_by_z(
                                massive_of_separated_ring_points(
                                    find_n_circle_1(pcd)[0]
                                        ,2)
                                    , -0.8)
                                    , 2)
    ff =[]
    x =[]
    y =[]
    x_d = []
    yg =[]
    for t in filtered:
        angle_1 = angle_of_point_by_ground(t)
        #print(angle_1)
        x.append(ori(t))
        y.append(t[2]*30)
        if 2.0 < angle_1 < 2.3 :
            ff.append(t)
            print(angle_of_point_by_ground(t))


            #filtered.index(t)
        #print(ff, 'was removed')
    #filtered = ff
    n_beams_filtered_cloud.points = Vector3dVector(filtered)
    ff = filtered
    it = 0
    ring_points_1 = []
    ra_d = []
    while it+1 < len(ff):
        first_proizv_dz_po_dy = math.sqrt((ff[it+1][2]-ff[it][2])**2) / (math.sqrt((ff[it+1][0]-ff[it][0])**2) + 0.00000001)
        first_proizv_dx_po_dy = math.sqrt((ff[it+1][0]-ff[it][0])**2) / (math.sqrt((ff[it+1][1]-ff[it][1])**2) + 0.00000001)
        x_d.append ( math.sqrt((ff[it+1][2]-ff[it][2])**2 / abs( ori(ff[it+1] - ff[it]))) )

        print(len(x_d))
        print (len(y))
        ra = math.sqrt((ff[it][0])**2+(ff[it][1])**2+(ff[it][2] -1.83 )**2)
        ra_d.append(ra*10)
##        x.append(ori(ff[it]))
##        y.append(ff[it][2])

        it +=1
        print (first_proizv_dz_po_dy)

        if (first_proizv_dx_po_dy < 2.5) and (first_proizv_dz_po_dy < 2.5 ) and (8.4 < ra < 8.7) :
                ring_points_1.append(ff[it])
                print('rad',ra)
    #filtered = ring_points_1
    zero_ring_cloud = PointCloud()
    zero_ring_cloud.points = Vector3dVector(filtered)
    #zero_ring_cloud = voxel_down_sample(zero_ring_cloud, voxel_size = 0.05)
    #draw_geometries([zero_ring_cloud])
    x_d.append(0)
    ra_d.append(0)
    rrr = 0
    zz=[]
    rasstoyanie_do_predidushei_tochki = 0
    spisok_tochek_filtrovannah_po_tangensu = []
    cut_mas = []
    clouds = []
    list_of_index_tang_sort = []
    while rrr+1 < len(y):
        yg.append ( abs(math.degrees( math.atan2(y[rrr+1] - y[rrr], x[rrr+1] - x[rrr]))) )
        if -30 < ( math.degrees( math.atan2(y[rrr+1] - y[rrr], x[rrr+1] - x[rrr])) ) < 30 :
            zz.append(ff[rrr])
            print(angle_of_point_by_ground(t))
        else:
            zz.append([ff[rrr][0], ff[rrr][1], 7])
            spisok_tochek_filtrovannah_po_tangensu.append([ff[rrr][0], ff[rrr][1], ff[rrr][2]])
            list_of_index_tang_sort.append(rrr)
            if len(spisok_tochek_filtrovannah_po_tangensu) > 1:
                #print(spisok_tochek_filtrovannah_po_tangensu)
                #print(spisok_tochek_filtrovannah_po_tangensu[len(spisok_tochek_filtrovannah_po_tangensu) - 1])
                dist = la.norm(np.array(spisok_tochek_filtrovannah_po_tangensu[- 1]) - np.array(spisok_tochek_filtrovannah_po_tangensu[- 2]))
                if dist > 1.775:
                    print ('!!!!!!!!!!!!!! i had find a corridor !!!!!!!!!!!!!!',dist)
                    print(list(ff[rrr]))
                    print(list(np.array(spisok_tochek_filtrovannah_po_tangensu[-1])) )
                    print ('nom right', rrr)
                    print ('nom left', list_of_index_tang_sort[-2])
                    cut_mas = zz[list_of_index_tang_sort[-2]:rrr]
                    clouds.append(cut_mas)
                    print('44444444444444444444444444!!!!!!!!!!!------------------',zz)
                    print('44444444444444444444444444!!!!!!!!!!!',cut_mas)
                    zz.append([spisok_tochek_filtrovannah_po_tangensu[- 1][0],spisok_tochek_filtrovannah_po_tangensu[- 1][1] , spisok_tochek_filtrovannah_po_tangensu[- 1][2]])
                    zz.append([spisok_tochek_filtrovannah_po_tangensu[- 2][0],spisok_tochek_filtrovannah_po_tangensu[- 2][1] , spisok_tochek_filtrovannah_po_tangensu[- 2][2]])

        rrr +=1
    print('______________list of indexes tan sort', list_of_index_tang_sort)
    #filtered = zz
    yg.append(0)


    print( ra_d)
    #print (y)
    #print(yg)
    plt.plot(x, y, linewidth=1)
    print (y)
    #plt.plot(x, x_d )
    plt.plot(x, yg )
    plt.plot(x, ra_d )
    plt.show()


    ring_cloud = PointCloud()
    ring_cloud.points = Vector3dVector(zz)
    #zero_ring_cloud = voxel_down_sample(zero_ring_cloud, voxel_size = 0.05)
    draw_geometries([ring_cloud])
    dummy = []
    pcd_tree = KDTreeFlann(ring_cloud)
#    one_plane_points = zz
    one_plane_points = spisok_tochek_filtrovannah_po_tangensu
    kolichestvo_tochek_v_okresnosti =[]
    ori_tochki = []
    for point_of in one_plane_points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_of, 2.5)
        rt = 0
        kolichestvo_tochek_v_okresnosti.append(k)
        ori_tochki.append(ori(point_of))
        if k > len(spisok_tochek_filtrovannah_po_tangensu)*0.165:
            dummy.append(point_of)

##        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_of, 2.5)
##        rt = 0
##        kolichestvo_tochek_v_okresnosti.append(k)
##        ori_tochki.append(ori(point_of))
##        if k > len(n_beams_filtered_cloud.points)*0.165:
##            dummy.append(point_of)
    #print(dummy)
    #plt.plot(ori_tochki, kolichestvo_tochek_v_okresnosti )
    #plt.show()
    dummy_cloud = PointCloud()
    dummy_cloud.points = Vector3dVector(clouds[0])
    n_beams_filtered_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    draw_geometries([dummy_cloud])

##    ## 2) Добавляем в облако точку 0
##
##    point_array_with_zero = np.insert(np.asarray(n_beams_filtered_cloud.points) , 0 , [0,0,0], axis=0)
##
##    b = point_array_with_zero

##    # RANSAC
##    ## Находим облако точек, лежащих в одной плоскости
##
##    max_iterations = int(len(n_beams_filtered_cloud.points)/100)
##    #max_iterations = 10
##    goal_inliers = len(n_beams_filtered_cloud.points)*0.8
##
##    m, one_plane_points  = run_ransac(n_beams_filtered_cloud.points, estimate, lambda x, y: is_inlier(x, y, 0.03), len(n_beams_filtered_cloud.points), goal_inliers, max_iterations)
##    a, b, c, d = m
##    sdup_cloud = PointCloud()
##    sdup_cloud.points = Vector3dVector(one_plane_points)
##    #draw_geometries([sdup_cloud])
##    yup = []
##    k_dot = 0
##    while k_dot < len(one_plane_points) - 1:
##        proizv = abs(one_plane_points[k_dot+1][0]-one_plane_points[k_dot][0])/abs(one_plane_points[k_dot+1][1]-one_plane_points[k_dot][1])
##        if 0 < proizv <= 2:
##            yup.append(one_plane_points[k_dot])
##        k_dot += 1
##    yup_cloud = PointCloud()
##    yup_cloud.points = Vector3dVector(yup)
##    pcd_tree = KDTreeFlann(sdup_cloud)
##    dummy = []
##    for point_of in one_plane_points:
##        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_of, 2.5)
##        rt = 0
##        if k > len(n_beams_filtered_cloud.points)*0.03:
##            dummy.append(point_of)
##    #print(dummy)
##    dummy_cloud = PointCloud()
##    dummy_cloud.points = Vector3dVector(dummy)
##    n_beams_filtered_cloud.paint_uniform_color([0.7, 0.7, 0.7])
##    pcd.paint_uniform_color([0.5, 0.5, 0.5])
##    draw_geometries([dummy_cloud])
##
##    lines_set = PointCloud()
##    lines_li = []
##    edge_point_array_left = []
##    edge_point_array_right = []
##    for i in find_n_circle_1(dummy_cloud)[1]:
##        super_cloud = massive_of_separated_ring_points(find_n_circle_1(dummy_cloud)[0], i)
##        mini = np.argmin((super_cloud), axis=0)[1]
##        maxi = np.argmax((super_cloud), axis=0)[1]
##        #print (mini)
##        #print (maxi)
##        edge_point_array_left.append(super_cloud[mini])
##        edge_point_array_right.append(super_cloud[maxi])
##        points_po = super_cloud
##        lines = [[mini, maxi]]
##        lines_li.append(lines)
##        shirina_dorogi = math.sqrt((super_cloud[mini][0] - super_cloud[maxi][0])**2+(super_cloud[mini][1] - super_cloud[maxi][1])**2+(super_cloud[mini][2] - super_cloud[maxi][2])**2)
##        #print (shirina_dorogi)
##
##    a_left , b_left, so_l = mnkGP(
##            list(np.asarray(edge_point_array_left)[:,0]),
##            list(np.asarray(edge_point_array_left)[:,1])
##            )
##    a_right , b_right, so_r = mnkGP(
##            list(np.asarray(edge_point_array_right)[:,0]),
##            list(np.asarray(edge_point_array_right)[:,1])
##            )
##    print ( 'ABS', abs((a_left) - (a_right)))
##    print (list(np.asarray(edge_point_array_left)[:,0]))
##    print (list(np.asarray(edge_point_array_left)[:,1]))
##
##    print ('w', list(np.asarray(edge_point_array_right)[:,0]))
##    print ('w', list(np.asarray(edge_point_array_right)[:,1]))
##
##
##    if ( a_right != 0 ) and  ( 0 < abs((a_left) - (a_right)) < 0.8 ) and ( abs(so_l) < 1.5 ) and ( abs (so_r) < 1.5 ):
##
##        points_su_left =[]
##        points_su_right =[]
##        for d_x in range ( 0 , 200 ):
##            line_1_x = d_x * 0.1
##            points_su_left.append( [ line_1_x , line_1_x*a_left + b_left, -1.83 ] )
##            points_su_right.append( [ line_1_x , line_1_x*a_right + b_right, -1.83 ] )
##
##
##
##
##        print(list(np.asarray(edge_point_array_left)[:,0]))
##        #dummy_cloud.points = Vector3dVector(super_cloud)
##        lines_set.points = Vector3dVector(edge_point_array_left + edge_point_array_right + points_su_left + points_su_right )
##        lines_set.paint_uniform_color([1, 0, 0])
##        #lines_set.lines = Vector2iVector(lines)
##        draw_geometries([ lines_set, yup_cloud, pcd])
##        print("--- %s seconds ---" % (time.time() - start_time))
##    else:
##        print('There is no road')
##        draw_geometries([ yup_cloud, pcd])














