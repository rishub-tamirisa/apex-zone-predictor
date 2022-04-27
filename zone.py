import cv2
from PIL import Image
import numpy as np

 
arr = []
rings = [1000/2, 550/2, 275/2, 150/2, 75/2, 0.05/2]

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
            arr.append([x,y])
            # print("first")
            # print(arr)

            img = cv2.circle(img, (x, y), int(rings[0] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), 2, (0,0,255),3)
            img = cv2.circle(img, (x, y), int(((rings[0] - rings[1]))*ppm), (20, 20, 20), 1)
            cv2.imshow('image', img)
        elif (len(arr) == 1):
            arr.append([x,y])
            # print("second")
            # print(arr)
            img = cv2.circle(img, (x, y), int(rings[1] * ppm), (255, 255, 255), 1)
            img = cv2.circle(img, (x, y), 2, (0,0,255),3)

            r3_pred = calc_zones(MAP_CENTER, arr[0][0], arr[0][1], x, y, 1)
            # arr.append([x,y])
            x3 = int(r3_pred[0])
            y3 = int(r3_pred[1])
            
            # print("third")
            arr.append([r3_pred[0],r3_pred[1]])
            # print(arr)
            img = cv2.circle(img, (x3, y3), int(rings[2] * ppm), (255, 0, 0), 2)
            img = cv2.circle(img, (x3, y3), 2, (0,0,255),3)

            #Attempts Zone 4 and 5 (Unstable)
            for i in range(2,4):
                # print(np.asarray([arr[i-2][0], arr[i-2][1]]))
                # print(arr[i-1])
                r_pred = calc_zones(np.asarray([arr[i-2][0], arr[i-2][1]]), arr[i-1][0], arr[i-1][1], arr[i][0], arr[i][1], i)
                x_r = int(r_pred[0])
                y_r = int(r_pred[1])
                arr.append([r_pred[0], r_pred[1]])
                # print(arr)

                img = cv2.circle(img, (x_r, y_r), int(rings[i+1] * ppm), (255, 255, 0), 2)
                img = cv2.circle(img, (x_r, y_r), 2, (0,0,255),3)

            # cv2.imwrite("WE-LAN-4-26-G1_Test.png", img)
            cv2.imshow('image', img)
        
        # if (len(arr) >= 2):
            # print(str(x) + " " + str(y))

count = 850
def calc_zones(center, x1, y1, x2, y2, ring):
    global count
    # print(x1)
    ring_1_center = np.array([x1,y1])
    ring_1_rad = rings[ring-1] * ppm
    # print("start")
    map_center = np.array(center)
    # print(map_center)
    # print(ring_1_center)
    r1_cen_to_mc = map_center - ring_1_center
    # print(r1_cen_to_mc)
    dist = np.linalg.norm(r1_cen_to_mc)
    # print(r1_cen_to_mc)
    rad_vec = None
    if dist != 0:
        rad_vec = ring_1_rad * (r1_cen_to_mc / dist)
    else:
        rad_vec = np.array([0,0])
    # print(rad_vec)
    V1 = None
    if dist <= ring_1_rad: # ring 1 contains center  
        V1 = r1_cen_to_mc - rad_vec
    else: # ring 1 outside center
        V1 = rad_vec - r1_cen_to_mc
    
    v1_norm = np.linalg.norm(V1)
    rad_norm = np.linalg.norm(rad_vec)
    if (rad_norm != 0):
        V1 = V1 * (1 - (v1_norm/rad_norm))

    ring_2_center = np.array([x2,y2])
    ring_2_rad = rings[ring] * ppm
    V2 = ring_2_center - ring_1_center
    V3 = None
    
    dist2 = np.linalg.norm(map_center - ring_2_center)
    if (dist2 <= ring_2_rad):
        V3 = V1 + V2 + map_center
    else:
        V3 = V1 + V2 + ring_1_center
    # print(ring_1_center)
    # print(ring_2_center)
    # print(V1)
    # print(V2)
    # print(V3)
    A = np.array([V1, V2])
    # print(V1)
    # print("RING SHIFT MATRIX:")
    # print(A)
    w, v = np.linalg.eig(A)
    msg = ""
    if isinstance(w[0], complex):
        msg = "Oscillatory Shift on Zone " + str(ring+2)
    else:
        msg = "Standard Shift on Zone " + str(ring+2)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,count)
    fontScale              = 0.4
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2

    cv2.putText(img,msg, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    count += 25
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
    # print(HEIGHT)
    # print(WIDTH)
    MAP_CENTER = [int(WIDTH / 2)-50, int(HEIGHT / 2)]
    
    ppm = HEIGHT / map_size
    
    img = cv2.circle(img, (MAP_CENTER[0], MAP_CENTER[1]), 3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()