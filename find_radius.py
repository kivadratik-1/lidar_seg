import numpy as np
import itertools

po_arr = (np.array([ 5.72452974, -2.34896803, -1.65799487]), np.array([ 6.97682333, -2.10111356, -1.68218398]), np.array([ 8.30490112, -1.64140236, -1.6455369 ]), np.array([ 9.65932083, -1.33003294, -1.5443213 ]), np.array([12.24297142, -1.00010204, -1.50825512]))
comb = itertools.combinations(po_arr, 3)
for e in comb:
    #print (e)
    A = e[0]
    B = e[1]
    C = e[2]

    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)

    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    print(R)
    #print(P)