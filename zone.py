import cv2
from PIL import Image
import numpy as np

 
arr = []
rings = [1000/2, 550/2, 300/2, 150/2, 75/2, 0.05/2]

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
            img = cv2.circle(img, (x, y), int(((rings[0] - rings[1]))*ppm), (20, 20, 20), 1)
            cv2.imshow('image', img)
        elif (len(arr) == 2):
            arr.append([x,y])

            #Draw Zone 2
            img = cv2.circle(img, (x, y), int(rings[1] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), 2, (0,0,255),3)

            color = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
            #Attempts Zone 3 | (4 & 5 unstable)
            for i in range(2,5):
                r_pred = calc_zones(np.asarray([arr[i-2][0], arr[i-2][1]]), arr[i-1][0], arr[i-1][1], arr[i][0], arr[i][1], i)
                x_r = int(r_pred[0])
                y_r = int(r_pred[1])
                arr.append([r_pred[0], r_pred[1]])
                # print(arr)
                img = cv2.circle(img, (x_r, y_r), int(rings[i] * ppm), color[i-2], 2)
                img = cv2.circle(img, (x_r, y_r), 2, (0,0,255),3)

            # cv2.imwrite("WE-LAN-4-26-G1_Test.png", img)
            cv2.imshow('image', img)

count = 850
def calc_zones(center, x1, y1, x2, y2, ring):
    global count
    ring_1_center = np.array([x1,y1])
    ring_1_rad = rings[ring-2] * ppm
    map_center = np.array(center)
    r1_cen_to_mc = map_center - ring_1_center
    dist = np.linalg.norm(r1_cen_to_mc)
    rad_vec = None
    if dist != 0:
        rad_vec = ring_1_rad * (r1_cen_to_mc / dist)
    else:
        rad_vec = np.array([0,0])
    V1 = None
    if dist <= ring_1_rad: # ring 1 contains center  
        V1 = r1_cen_to_mc - rad_vec
    else: # ring 1 outside center
        V1 = rad_vec - r1_cen_to_mc
    
    v1_norm = np.linalg.norm(V1)
    rad_norm = np.linalg.norm(r1_cen_to_mc)
    # if (rad_norm != 0):
        # V1 = V1 * (1 - (v1_norm/rad_norm))

    ring_2_center = np.array([x2,y2])
    ring_2_rad = rings[ring-1] * ppm
    V2 = ring_2_center - ring_1_center
    V3 = None
    
    dist2 = np.linalg.norm(map_center - ring_2_center)
    if (dist2 <= dist):
        V3 = V1 + V2 + map_center
    else:
        V3 = V1 + V2 + ring_1_center
    # V3 = V1 + V2 + ring_1_center
    # print(V1)
    # print(V2)
    # print(V3)
    A = np.array([V1, V2])
    w, v = np.linalg.eig(A)
    msg = ""
    if isinstance(w[0], complex):
        msg = "Oscillatory Shift on Zone " + str(ring+1)
    else:
        msg = "Standard Shift on Zone " + str(ring+1)


    cv2.putText(img,msg, (10,count), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 2)
    count += 25
    max_shift = ((rings[ring-1] - rings[ring]))*ppm
    shift = np.linalg.norm(V3 - ring_2_center)
    
    if (shift > max_shift):
        print("here on zone: " + str(ring+1))
        V3 = ring_2_center + max_shift * (V3 - ring_2_center)/(np.linalg.norm(V3 - ring_2_center))
    shift = np.linalg.norm(V3 - ring_2_center)
    # print(shift)
    # print("Eigenvalue/vectors:")
    # print(w)
    # print(v)
    return V3

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