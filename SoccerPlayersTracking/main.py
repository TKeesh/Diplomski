import numpy as np
import cv2, time


def define_rect(image, win_name):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """
    clone = image.copy()
    rect_pts = [] # Starting and ending points
    #win_name = "image" # Window name

    def select_points(event, x, y, flags, param):
        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 1)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()
        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return np.array(rect_pts, np.int32)


def get_dominant_colors(img, K=2):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow('K=%s'%str(K),res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_field_mask_smartly(videoPath):
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    points = define_rect(frame, "Field selection")
    print (points)

    x0, y0 = points[0][0], points[0][1]
    x1, y1 = points[1][0], points[1][1]
    rect = frame[y0:y1, x0:x1]

    avg_color, std_color = cv2.meanStdDev(rect)
    print (avg_color, std_color)
    
    fieldMask = cv2.inRange(frame, avg_color - std_color*2, avg_color + std_color*2)
    fieldMasked = cv2.bitwise_and(frame, frame, mask = fieldMask)
    cv2.imshow("Field masked", np.hstack([frame, fieldMasked]))
    # get_dominant_colors(rect, 1)
    # get_dominant_colors(rect, 2)
    # get_dominant_colors(rect, 3)
    # get_dominant_colors(rect, 4)
    # get_dominant_colors(rect, 5)
    # get_dominant_colors(rect, 6)
    # get_dominant_colors(rect, 7)
    cv2.waitKey(0)
    cv2.destroyWindow("Field masked")
    cap.release()

    return None, points


def define_poly(image, win_name):
    clone = image.copy()
    rect_pts = [] # Starting and ending points

    def select_points(event, x, y, flags, param):
        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts.append((x, y))            
            if len(rect_pts) == 1: 
                cv2.circle(clone, (x,y), 4, (0,0,255), -1)
                cv2.circle(clone, (x,y), 12, (0,0,255), 1)
            elif np.linalg.norm(np.array(rect_pts[-1]) - np.array(rect_pts[0])) < 12:                    
                cv2.line(clone, rect_pts[-2], rect_pts[0], (0,0,255), 1)
                rect_pts.pop(-1)
                #cv2.polylines(clone, [np.array(rect_pts, np.int32)], True, (0, 255, 0), 1)
            else:
                cv2.circle(clone, (x,y), 4, (0,0,255), -1)
                cv2.line(clone, rect_pts[-2], rect_pts[-1], (0,0,255), 1)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:       
        cv2.imshow(win_name, clone) # display the image and wait for a keypress
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image+            
            clone = image.copy()
            rect_pts = []
        elif key == ord("c"): # Hit 'c' to confirm the selection
            break
        elif key == ord("x"):
            rect_pts.pop(-1)
            clone = image.copy()
            for i, xy in enumerate(rect_pts):
                cv2.circle(clone, xy, 4, (0,0,255), -1)
                if not i:
                    cv2.circle(clone, xy, 12, (0,0,255), 1)
                if i + 1 == len(rect_pts): break
                cv2.line(clone, xy, rect_pts[i+1], (0,0,255), 1)
            cv2.imshow(win_name, clone)

    cv2.destroyWindow(win_name)

    return np.array(rect_pts, np.int32)


def get_field_mask_brute(videoPath):
    cap = cv2.VideoCapture(videoPath)    
    ret, frame = cap.read()
    points = define_poly(frame, "Field selection")
    print (points)

    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    fieldMasked = cv2.bitwise_and(frame, frame, mask = mask)

    # avg_color, std_color = cv2.meanStdDev(frame, mask=mask)
    # print (avg_color, std_color)
    
    # cv2.imshow("Field masked", np.hstack([fieldMasked, frame]))
    # cv2.waitKey(0)
    # cv2.destroyWindow("Field masked")
    cap.release()

    return mask, points


# def get_canny(img, sigma=0.33):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     v = np.median(gray)
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(gray, lower, upper)
#     cv2.imshow('Edges',edged)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return edged

def get_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh   
    edged = cv2.Canny(gray, lowThresh, high_thresh)
    cv2.imshow('Edges',edged)
    return edged


def init_bckg_histogram(frame):
    h, w, c = frame.shape
    return [[[{} for k in range(3)] for j in range(w)] for i in range(h)]


def update_bckg_histogram(bckg_hist, frame):
    h, w, c = frame.shape
    for i in range(h):
        for j in range(w):
            gbr = frame[i,j]
            if gbr[0] not in bckg_hist[i][j][0].keys(): bckg_hist[i][j][0][gbr[0]] = 1
            else: bckg_hist[i][j][0][gbr[0]] += 1
            if gbr[1] not in bckg_hist[i][j][1].keys(): bckg_hist[i][j][1][gbr[1]] = 1
            else: bckg_hist[i][j][1][gbr[1]] += 1
            if gbr[2] not in bckg_hist[i][j][2].keys(): bckg_hist[i][j][2][gbr[2]] = 1
            else: bckg_hist[i][j][2][gbr[2]] += 1
            

def get_bckg_histogram(backgrounds):
    #backgrounds = np.array(backgrounds)
    h = len(backgrounds[0])
    w = len(backgrounds[0][0])
    c = len(backgrounds[0][0][0])
    bckg_hist = [[[{} for k in range(c)] for j in range(w)] for i in range(h)]
    print(h,w,c)
    print(len(bckg_hist))
    print(len(bckg_hist[0]))
    print(len(bckg_hist[0][0]))
    for i in range(h):
        for j in range(w):
            for k in range(c):
                unique, counts = np.unique(backgrounds[:,i,j,k], return_counts=True)                
                bckg_hist[i][j][k] = dict(zip(unique, counts))
    print (bckg_hist[0][0][0])
    quit()


def get_background(videoPath):
    cap = cv2.VideoCapture(videoPath)
    step = 30#30
    count = 30#30
    ret, frame = cap.read() 
    h, w, c = frame.shape  
    backgrounds = np.array((count,h,w,c))
    bckg_count = init_bckg_histogram(frame)
    for i in range(count):
        for j in range(step):
            ret, frame = cap.read()
        background = cv2.GaussianBlur(frame,(5,5),0)
        backgrounds[i] = background
        #update_bckg_histogram(bckg_count, background)
    #print (bckg_count[0][0][0])

    get_bckg_histogram(backgrounds)



def get_background_sequenced(videoPath, fgbg):
    cv2.namedWindow('bckg_generator')
    cap = cv2.VideoCapture(videoPath)
    step = 500
    count = 10
    backgrounds = []
    for i in range(count):
        for j in range(step):
            ret, frame = cap.read()
        fgbg.apply(frame, learningRate=-1)
        cv2.imshow('bckg_generator', frame)
        cv2.waitKey(1)
        #background = cv2.GaussianBlur(frame,(5,5),0)
        #backgrounds.append(background)
    cv2.destroyWindow('bckg_generator')


def main(videoPath):
    cv2.ocl.setUseOpenCL(True)
    #fieldMask =  get_field_mask_smartly(videoPath)
    #quit()
    fieldMask, maskPoints = get_field_mask_brute(videoPath)
    #background = get_background(videoPath)

    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    background = get_background_sequenced(videoPath, fgbg)

    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    print (frame.shape)
    # print (frame[-1].shape)
    # print (frame[0].shape)
    # print (frame[0][0].shape)
    #quit()
    i = -1
    while(True):
        # Capture frame-by-frame
        i += 1
        #timeS = time.time()
        ret, frame = cap.read()
        if i % 10: 
            continue

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.bitwise_and(frame, frame, mask = fieldMask)
        #cnyImg = get_canny(frame)

        fgmask = fgbg.apply(frame, learningRate=0.01)
        bckg = fgbg.getBackgroundImage()
        
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask = fieldMask)
        # Display the resulting frame
        cv2.imshow('frame', fgmask)
        #print (time.time() - timeS)
        if cv2.waitKey(1) & 0xFF == ord('q'):            
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    #main('../Dinamo_vs._Rijeka_-_panorama_view.mp4')
    #main('../Inter - Dinamo wide angle.mp4')
    main('../probaStitching_20000.avi')