import cv2
from PIL import Image
import numpy as np

 
arr = []
rings = [1000/2, 550/2, 400/2, 200/2, 100/2, 0.05/2]

HEIGHT = None
WIDTH = None

MAP_CENTER = [None, None]
img = None

map_size = 1925
ppm = None

def click_event(event, x, y, flags, params):
    global img
    if event == cv2.EVENT_LBUTTONDOWN and len(arr) < 2:
        arr.append([x,y])
        
        if len(arr) == 1:
            # print(int(rings[0] * ppm/2))
            img = cv2.circle(img, (x, y), int(rings[0] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), int(((rings[0] - rings[1]))*ppm), (20, 20, 20), 1)
        if (len(arr) == 2):
            img = cv2.circle(img, (x, y), int(rings[1] * ppm), (255, 255, 255), 1)
            # print(int(rings[1] * ppm/2))
            print(arr)
            print("arr^^")
            r3_pred = calc_zones(arr[0][0], arr[0][1], arr[1][0], arr[1][1])
            x3 = int(r3_pred[0])
            y3 = int(r3_pred[1])
            img = cv2.circle(img, (x3, y3), 10, (255, 255, 255), 1)
            cv2.imwrite("WE-LAN-4-26-G1_Test.png", img)
        cv2.imshow('image', img)

def calc_zones(x1, y1, x2, y2):
    # print(x1)
    ring_1_center = np.array([x1,y1])
    ring_1_rad = rings[0] * ppm
    print("start")
    map_center = np.array(MAP_CENTER)
    print(map_center)
    r1_cen_to_mc = map_center - ring_1_center
    print(r1_cen_to_mc)
    dist = np.linalg.norm(r1_cen_to_mc)
    rad_vec = ring_1_rad * (r1_cen_to_mc / dist)
    print(rad_vec)
    V1 = None
    if dist <= ring_1_rad: # ring 1 contains center  
        V1 = r1_cen_to_mc - rad_vec
    else: # ring 1 outside center
        V1 = rad_vec - r1_cen_to_mc
    
    v1_norm = np.linalg.norm(V1)
    rad_norm = np.linalg.norm(ring_1_rad)
    V1 = V1 * (1 - (v1_norm/rad_norm))

    ring_2_center = np.array([x2,y2])
    V2 = ring_2_center - ring_1_center
    V3 = V1 + V2 + ring_1_center
    print(ring_1_center)
    print(ring_2_center)
    print(V1)
    print(V2)
    print(V3)
    return V3


if __name__=="__main__":
    filepath = "worlds-edge.png"
    img = cv2.imread(filepath, 1)
    height, width, _ = img.shape
    img = cv2.resize(img, [height, height])
    HEIGHT, WIDTH, channels = img.shape
    print(HEIGHT)
    print(WIDTH)
    MAP_CENTER = [int(WIDTH / 2)-50, int(HEIGHT / 2)]
    
    ppm = HEIGHT / map_size
    
    img = cv2.circle(img, (MAP_CENTER[0], MAP_CENTER[1]), 3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()