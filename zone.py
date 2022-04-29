import cv2
from PIL import Image
import numpy as np
from math import cos, sin

#Global Constants 
arr = []
rings = [1000/2, 525/2, 300/2, 150/2, 75/2, 0.05/2]

 #Counterpull Strength

HEIGHT = None
WIDTH = None

MAP_CENTER = [None, None]
img = None

map_size = 1925
ppm = None

def click_event(event, x, y, flags, params):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if len(arr) == 0:
            arr.append(MAP_CENTER)
            arr.append([x,y])

            #Draw Zone 1
            img = cv2.circle(img, (x, y), int(rings[0] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), 2, (0,0,255),3)

            #Draw Zone 2 Center Guide
            img = cv2.circle(img, (x, y), int(((rings[0] - rings[1]))*ppm), (100,100,100), 1)
            cv2.imshow('image', img)
        elif (len(arr) == 2):
            arr.append([x,y])

            #Draw Zone 2
            img = cv2.circle(img, (x, y), int(rings[1] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), 2, (0,0,255),3)

            color = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
            #Attempts Zone 3 | (4 & 5 unstable)
            for i in range(2,5):
                r_pred = calc_zones(np.asarray([arr[i-2][0], arr[i-2][1]]), arr[i-1][0], arr[i-1][1], arr[i][0], arr[i][1], i, False)
                x_r = int(r_pred[0])
                y_r = int(r_pred[1])
                arr.append([r_pred[0], r_pred[1]])
                # print(arr)
                img = cv2.circle(img, (x_r, y_r), int(rings[i] * ppm), color[i-2], 2)
                img = cv2.circle(img, (x_r, y_r), 2, (0,0,255),3)

            # cv2.imwrite("WE-LAN-4-26-G1_Test.png", img)
            cv2.imshow('image', img)

def compute_vecs(center, x1,y1,ring):
    ring_1_center = np.array([x1,y1])
    ring_1_rad = rings[ring] * ppm
    r1_cen_to_mc = center - ring_1_center
    return ring_1_center, ring_1_rad, r1_cen_to_mc

count = 850

def calc_zones(center, x1, y1, x2, y2, ring, pull):
    global count

    map_center = np.array(center) #center of previous ring
    ring_1_center, ring_1_rad, r1_cen_to_mc = compute_vecs(map_center, x1, y1, ring-2)
    dist = np.linalg.norm(r1_cen_to_mc) #distance from ring center to previous center

    rad_vec = None
    if dist != 0:
        rad_vec = ring_1_rad * (r1_cen_to_mc / dist) #vector pointing from center of ring to edge with length radius
    else:
        rad_vec = np.array([0,0]) # define 0 vector

    #Compute V1 by subtracting vector from center to ring from radius vector
    V1 = None
    if dist <= ring_1_rad: # ring 1 contains center  
        V1 = r1_cen_to_mc - rad_vec
    else: # ring 1 outside center
        V1 = rad_vec - r1_cen_to_mc
    
    ring_2_center, ring_2_rad, _ = compute_vecs(map_center, x2, y2, ring-1)

    #compute V2 by subtracting ring 2 center and ring 1 center
    V2 = ring_2_center - ring_1_center
    V3 = None
    
    dist2 = np.linalg.norm(map_center - ring_2_center)
    if (dist2 <= dist):
        V3 = V1 + V2 + map_center
    else:
        V3 = V1 + V2 + ring_1_center
    
    #TODO: Implement counterpulls with key presses
    if (np.linalg.norm(r1_cen_to_mc) != 0):
        if pull:
            print("counter at ring: " + str(ring+1))
            print("V3 before: " + str(V3 - ring_1_center))
            V3 = calc_counter(V1, V2, ring_1_center, r1_cen_to_mc, 2)
            print("V3 after: " + str(V3 - ring_1_center))

    A = np.array([V1, V2])
    w, v = np.linalg.eig(A)
    msg = ""
    if isinstance(w[0], complex):
        msg = "Oscillatory Shift on Zone " + str(ring+1)
    else:
        msg = "Standard Shift on Zone " + str(ring+1)

    printCV(msg, (255,255,255))

    max_shift = ((rings[ring-1] - rings[ring]))*ppm
    shift = np.linalg.norm(V3 - ring_2_center)
    if (shift > max_shift):
        V3 = closestPtOnCircle(ring_2_center, max_shift, V3)
        
    shift = np.linalg.norm(V3 - ring_2_center)
    # print(shift)
    
    return V3

PULL_MULTIPLIER = 1.0
def calc_counter(V1, V2, center, rad, m):
    global PULL_MULTIPLIER

    V3 = V1 + V2
    print("V3 local : " + str(V3))
    theta1 = np.arctan(V1[1]/V1[0])
    theta2 = np.arctan(V2[1]/V2[0])
    print("Angle: " + str(theta1 * 360/(2*np.pi)))
    print("Angle: " + str(theta2 * 360/(2*np.pi)))
    theta = (theta2 - theta1) / 1.2
    theta = theta * PULL_MULTIPLIER
    PULL_MULTIPLIER = PULL_MULTIPLIER * 0.8
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    counter = rot.dot(V3)
    counterpull_fac1 = V1 / np.linalg.norm(V1) * 6
    counterpull_fac2 = V2 / np.linalg.norm(V2) * 6

    print( "V: " + str(counter))
    V =  (counter + center + counterpull_fac1 + counterpull_fac2)
    return V#(m - m*(1-PULL_STRENGTH))*V2 - counterpull_fac + center

def printCV(msg, color):
    global count
    cv2.putText(img,msg, (10,count), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, 2)
    count += 25

def closestPtOnCircle(center, radius, vec):
    return center + radius * (vec - center)/(np.linalg.norm(vec - center))

if __name__=="__main__":
    filepath = "worlds-edge.png"
    img = cv2.imread(filepath, 1)
    height, width, _ = img.shape
    img = cv2.resize(img, [height, height])
    HEIGHT, WIDTH, channels = img.shape
    MAP_CENTER = [int(WIDTH / 2)-50, int(HEIGHT / 2)]
    
    ppm = HEIGHT / map_size
    
    img = cv2.circle(img, (MAP_CENTER[0], MAP_CENTER[1]), 3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()