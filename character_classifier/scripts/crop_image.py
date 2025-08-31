import cv2
import numpy as np

def get_thresholded_image(image):
    
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    edge_image = cv2.convertScaleAbs( cv2.magnitude(sobelx,sobely) )
    _, thresholded_image = cv2.threshold(edge_image, 90, 255, type=cv2.THRESH_BINARY)
    
    return thresholded_image

def get_crop_dimensions(image):
    '''
    
        IMAGE: A NumPy array with shape `(height, width, channels=3)`
            
        RETURNS: Tuple in the form `(y_top, y_bottom, x_left, x_right)` as the bounds for 
        a cropped version of the image
   
    '''    
    
    # find edges and apply threshold
    image_thresholded = get_thresholded_image(image)
    
    # define constants required for large bounding box
    max_y, max_x, _ = np.shape(image)
    x_left = -1
    y_top = -1
    x_right = -1
    y_bottom = -1
    
    # determine contours in the image and draw boxes around each
    image_gray = cv2.cvtColor(image_thresholded, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 80:
            x, y, w, h = cv2.boundingRect(contour)
            ## check for updates in perimeter
            if x_left == -1 or x < x_left:
                x_left = x
            if x_right == -1 or x+w > x_right:
                x_right = x+w
            if y_top == -1 or y < y_top:
                y_top = y
            if y_bottom == -1 or y+h > y_bottom:
                y_bottom = y+h
            cv2.rectangle(image_thresholded, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # add a small padding around the bounding box area
    x_left = x_left-10     if x_left-10 >= 0 else 0
    x_right = x_right+10   if x_right+10 <= max_x else max_x
    y_top = y_top-10       if y_top-10 >= 0 else 0
    y_bottom = y_bottom+10 if y_bottom+10 <= max_y else max_y
    
    # draw large bounding box
    cv2.rectangle(image_thresholded, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)
    
    return y_top, y_bottom, x_left, x_right

def crop_image(image, thresholded=False):
    '''
    
        IMAGE: A NumPy array with shape `(height, width, channels=3)`
        
        THRESHOLDED: Set to True to return the thresholded version with contours shown,
        else the (cropped) original image will be returned
        
        RETURNS: The cropped image, depending on the value of `thresholded` parameter
   
    '''    

    y_top, y_bottom, x_left, x_right = get_crop_dimensions(image)
    
    if (thresholded):
        thresholded_image = get_thresholded_image(image)
        return thresholded_image[y_top:y_bottom, x_left:x_right]
    else:
        return image[y_top:y_bottom, x_left:x_right]

if __name__=="__main__":
    # replace with path to any image file
    images = [ # assuming file run from monorepo (using VS code, usually the case for name=main)
        cv2.imread('./character_classifier/custom_test_images/IMG_1949.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1949-2.jpg'),

        cv2.imread('./character_classifier/custom_test_images/IMG_2000.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2001.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2001-2.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2002.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2003.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2004.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2005.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2006.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2007.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2008.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2009.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2010.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2011.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2012.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2013.jpg'), 
        cv2.imread('./character_classifier/custom_test_images/IMG_2014.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2015.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2016.jpg'),
    
        cv2.imread('./character_classifier/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1977.jpg'),
        
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0001.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0002.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0003.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0004.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0005.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0006.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0007.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0008.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0009.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/书/0010.png', np.uint8), cv2.IMREAD_UNCHANGED),
    ]
        
    # image = cv2.imread('./character_classifier/custom_test_images/IMG_1976.jpg')
    for image in images:
        
        ## add channels dimension
        if (len(np.shape(image)) == 2):
            image = image[..., np.newaxis]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
