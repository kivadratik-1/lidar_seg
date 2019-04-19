import numpy as np
from numpy import array, abs as np_abs
from open3d import *
from numpy.fft import rfft, hfft, ihfft
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
#from svd_solve import svd, svd_solve
from ransac import *
import os
import matplotlib
import time

import scipy as sp
import matplotlib.pyplot as plt

start_time = time.time()


def find_n_circle(point_cloud):
    ## Функция разбивает поинт клауд на 16 колец (n_beams), возвращает точки с id кольца
    startOri = -1 * math.atan2(point_cloud.points[0][1], point_cloud.points[0][0])
    endOri = -1 * math.atan2(point_cloud.points[(len(point_cloud.points) - 1)][1], point_cloud.points[(len(point_cloud.points) - 1)][0]) + 2 * math.pi;
    count = len(point_cloud.points)
    separated_point_cloud = []
    scans_ids = []
    n_beams = 16
    if (endOri - startOri > 3 * math.pi):
        endOri -= 2 * math.pi
    elif endOri - startOri < math.pi:
        endOri += 2 * math.pi
    for one_point in point_cloud.points:
        #print(one_point)
        angle = math.atan(one_point[2] / math.sqrt(one_point[0] * one_point[0] + one_point[1] * one_point[1])) * 180 / math.pi
        scanID = 0
        scanID = int((angle + 15) / 2 + 0.5)
        scans_ids.append(scanID)
        separated_point_cloud.append([one_point, scanID])
    return separated_point_cloud

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
        first_proizv_dz_po_dy = math.sqrt((ring_points[it+1][2]-ring_points[it][2])**2) / (math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) + 0.00000001)
        first_proizv_dx_po_dy = math.sqrt((ring_points[it+1][0]-ring_points[it][0])**2) / (math.sqrt((ring_points[it+1][1]-ring_points[it][1])**2) + 0.00000001)
        ra = math.sqrt((ring_points[it][0])**2+(ring_points[it][1])**2+(ring_points[it][2] -1.83 )**2)
        it +=1
        if n_ring == 0 :
            if (6.5 < ra < 9.) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 1 :
            if (7.8 < ra < 9.5) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 2 :
            if (7.8 < ra < 10.5) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 3 :
            if (7.8 < ra < 12.5) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
        elif n_ring == 4 :
            if (11.5 < ra < 14.5) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
                ring_points_1.append(ring_points[it])
##        elif n_ring == 5 :
##            if (17 < ra < 21) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
##                ring_points_1.append(ring_points[it])
##        elif n_ring == 6 :
##            if (7.8 < ra < 20) and (first_proizv_dx_po_dy < 1) and (first_proizv_dz_po_dy < 0.5 ):
##                ring_points_1.append(ring_points[it])
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
    plt.plot(x, y, 'o', label='Original data', markersize=2)
    plt.plot(fx, f(fx), linewidth=1)
    plt.grid(True)
    plt.show()
    return fp[0] , fp[1] , so


if __name__ == '__main__':

    file = '4.pcd'

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud('D:/Dropbox/Inno/Inno/roadedges/'+file)
    draw_geometries([pcd])



    ## Part of Roma's code to detect rings:
    ## 1) Разбираем облако точек на 16 лучей

    n_beams_filtered_cloud = PointCloud()
    filtered = []
    for i in range(0,6):
        filtered = filtered + filtering_by_rings(cutoff_by_z(massive_of_separated_ring_points(find_n_circle(pcd),i), -1.3) , i)

    n_beams_filtered_cloud.points = Vector3dVector(filtered)

##
##    zero_ring_points = filtering_by_rings(
##                            cutoff_by_z(
##                                massive_of_separated_ring_points(
##                                    find_n_circle(pcd),0
##                                                                ), -1.5
##                                        ) , 0
##                                            )
##
##    #print(zero_ring_points)
##
##
##
##
    zero_ring_cloud = PointCloud()
    zero_ring_cloud.points = Vector3dVector(filtered)
    draw_geometries([zero_ring_cloud])


    ## 2) Добавляем в облако точку 0

    point_array_with_zero = np.insert(np.asarray(n_beams_filtered_cloud.points) , 0 , [0,0,0], axis=0)

    b = point_array_with_zero

    # RANSAC
    ## Находим облако точек, лежащих в одной плоскости

    max_iterations = int(len(n_beams_filtered_cloud.points)/100)
    goal_inliers = len(n_beams_filtered_cloud.points)*0.8

    m, one_plane_points  = run_ransac(n_beams_filtered_cloud.points, estimate, lambda x, y: is_inlier(x, y, 0.03), len(n_beams_filtered_cloud.points), goal_inliers, max_iterations)
    a, b, c, d = m
    sdup_cloud = PointCloud()
    sdup_cloud.points = Vector3dVector(one_plane_points)
    #draw_geometries([sdup_cloud])
    yup = []
    k_dot = 0
    while k_dot < len(one_plane_points) - 1:
        proizv = abs(one_plane_points[k_dot+1][0]-one_plane_points[k_dot][0])/abs(one_plane_points[k_dot+1][1]-one_plane_points[k_dot][1])
        if 0 < proizv <= 2:
            yup.append(one_plane_points[k_dot])
        k_dot += 1
    yup_cloud = PointCloud()
    yup_cloud.points = Vector3dVector(yup)
    pcd_tree = KDTreeFlann(sdup_cloud)
    dummy = []
    for point_of in one_plane_points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_of, 2.5)
        rt = 0
        if k > len(n_beams_filtered_cloud.points)*0.03:
            dummy.append(point_of)
    #print(dummy)
    dummy_cloud = PointCloud()
    dummy_cloud.points = Vector3dVector(dummy)
    n_beams_filtered_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
   #draw_geometries([dummy_cloud])

    lines_set = PointCloud()
    lines_li = []
    edge_point_array_left = []
    edge_point_array_right = []
    for i in range(0,5):
        super_cloud = massive_of_separated_ring_points(find_n_circle(dummy_cloud),i)
        mini = np.argmin((super_cloud), axis=0)[1]
        maxi = np.argmax((super_cloud), axis=0)[1]
        #print (mini)
        #print (maxi)
        edge_point_array_left.append(super_cloud[mini])
        edge_point_array_right.append(super_cloud[maxi])
        points_po = super_cloud
        lines = [[mini, maxi]]
        lines_li.append(lines)
        shirina_dorogi = math.sqrt((super_cloud[mini][0] - super_cloud[maxi][0])**2+(super_cloud[mini][1] - super_cloud[maxi][1])**2+(super_cloud[mini][2] - super_cloud[maxi][2])**2)
        print (shirina_dorogi)

    a_left , b_left, so_l = mnkGP(
            list(np.asarray(edge_point_array_left)[:,0]),
            list(np.asarray(edge_point_array_left)[:,1])
            )
    a_right , b_right, so_r = mnkGP(
            list(np.asarray(edge_point_array_right)[:,0]),
            list(np.asarray(edge_point_array_right)[:,1])
            )
    points_su_left =[]
    points_su_right =[]
    for d_x in range ( 0 , 200 ):
        line_1_x = d_x * 0.1
        points_su_left.append( [ line_1_x , line_1_x*a_left + b_left, -1.83 ] )
        points_su_right.append( [ line_1_x , line_1_x*a_right + b_right, -1.83 ] )




    print(list(np.asarray(edge_point_array_left)[:,0]))
    #dummy_cloud.points = Vector3dVector(super_cloud)
    lines_set.points = Vector3dVector(edge_point_array_left + edge_point_array_right + points_su_left + points_su_right )
    lines_set.paint_uniform_color([1, 0, 0])
    #lines_set.lines = Vector2iVector(lines)
    draw_geometries([ lines_set, pcd])
    print("--- %s seconds ---" % (time.time() - start_time))














