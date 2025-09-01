import cv2
import numpy as np

def get_thresholded_image(image):
    
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    edge_image = cv2.convertScaleAbs( cv2.magnitude(sobelx,sobely) )
    _, thresholded_image = cv2.threshold(edge_image, 90, 255, type=cv2.THRESH_BINARY)
    
    return thresholded_image

def get_crop_dimensions(image):
    """
    Calculates crop dimensions based on visual contours of the image.
    This also automatically blacks out small contours on the returned
    thresholded image. No effect on the input image

    Args:
        image (NumPy Array): A NumPy array with shape `(height, width, channels=3)`
    
    Returns: 
        Tuple in the form `(y_top, y_bottom, x_left, x_right, thres_img)` as the 
        bounds for a cropped version of the image, along with the uncropped thresholded image
    """

        

    # find edges and apply threshold
    image_thresholded = get_thresholded_image(image)
    
    # define constants required for large bounding box
    max_y, max_x, _ = np.shape(image)
    img_area = max_y * max_x
    x_left = -1
    y_top = -1
    x_right = -1
    y_bottom = -1
    
    # determine contours in the image and draw boxes around each
    image_gray = cv2.cvtColor(image_thresholded, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > img_area*0.00005:
            ## check for updates in perimeter
            if x_left == -1 or x < x_left:
                x_left = x
            if x_right == -1 or x+w > x_right:
                x_right = x+w
            if y_top == -1 or y < y_top:
                y_top = y
            if y_bottom == -1 or y+h > y_bottom:
                y_bottom = y+h
                
            ### draw a green box around the contour
            # # cv2.rectangle(image_thresholded, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # black out small contours on thresholded image
            cv2.rectangle(image_thresholded, (x, y), (x+w, y+h), (0, 0, 0), cv2.FILLED)
            
    # add a small padding around the bounding box area
    x_left = x_left-10     if x_left-10 >= 0 else 0
    x_right = x_right+10   if x_right+10 <= max_x else max_x
    y_top = y_top-10       if y_top-10 >= 0 else 0
    y_bottom = y_bottom+10 if y_bottom+10 <= max_y else max_y
    
    # draw large bounding box in blue
    # cv2.rectangle(image_thresholded, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)
    
    return y_top, y_bottom, x_left, x_right, image_thresholded

def crop_image(image, thresholded=False):
    """
    Args:
        image (NumPy Array): A NumPy array with shape `(height, width, channels=3)`
        
        thresholded (bool, optional): Set to True to use the thresholded version of the provided `image`, which also includes small contour black-outs. Defaults to False.

    Returns:
        (NumPy Array): NumPy array with shape `(height, width, channels=3)` representing the
        cropped `image`, depending on the value of `thresholded` parameter
    """
    
    y_top, y_bottom, x_left, x_right, thresholded_image = get_crop_dimensions(image)

    if (thresholded):
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
        
        cv2.imread('./character_classifier/custom_test_images/IMG_2100.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_2101.jpg'),
        
        cv2.imread('./character_classifier/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./character_classifier/custom_test_images/IMG_1977.jpg'),
        
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/我/0001.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/我/0002.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/我/0003.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/我/0004.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/我/0005.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0006.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0007.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0008.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0009.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./character_classifier/data/filtered/top-500/train/回/0010.png', np.uint8), cv2.IMREAD_UNCHANGED),
    ]
        
    for image in images:
        
        ## add channels dimension
        if (len(np.shape(image)) == 2):
            image = image[..., np.newaxis]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        
        # # cropped = crop_image(image, thresholded=True)
        # # grayscale_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # # grayscale_cropped_resized = cv2.resize(grayscale_cropped, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        # # cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        # # cv2.imshow("Cropped", cropped)
        # # cv2.imshow("Grayscale", grayscale_cropped)
        # # cv2.imshow("Resized", grayscale_cropped_resized)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()