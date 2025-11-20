import cv2
import numpy as np    
    
def convert_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (cv2.mean(thresholded)[0] < 100):
        thresholded = cv2.bitwise_not(thresholded)
        
    return thresholded
    
if __name__ == "__main__":
    images = [ # assuming file run from monorepo (using VS code, usually the case for name=main)
        # typed fonts, black text & white background
        cv2.imread('./model/custom_test_images/IMG_1949.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1949-2.jpg'),

        # computer drawn, black text & white background
        cv2.imread('./model/custom_test_images/IMG_2000.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2001.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2001-2.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2002.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2003.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2004.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2005.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2006.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2007.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2008.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2009.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2010.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2011.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2012.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2013.jpg'), 
        cv2.imread('./model/custom_test_images/IMG_2014.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2015.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2016.jpg'),
        
        # computer drawn, non-BW background / foreground colors
        cv2.imread('./model/custom_test_images/IMG_2100.jpg'),
        cv2.imread('./model/custom_test_images/IMG_2101.jpg'),
        
        # hand drawn
        cv2.imread('./model/custom_test_images/IMG_1975.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1976.jpg'),
        cv2.imread('./model/custom_test_images/IMG_1977.jpg'),
        
        # actual training samples
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/十/0001.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/只/0002.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/回/0003.png', np.uint8), cv2.IMREAD_UNCHANGED),
        cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/教/0004.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0005.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0006.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0007.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0008.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0009.png', np.uint8), cv2.IMREAD_UNCHANGED),
        # cv2.imdecode(np.fromfile('./model/data/filtered/top-500/train/书/0010.png', np.uint8), cv2.IMREAD_UNCHANGED),
    ]
    

    for image in images: 
        ## add channels dimension if not present
        if (len(np.shape(image)) == 2):
            image = image[..., np.newaxis]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        thresholded = convert_image(image)
        
        cv2.imshow("image", thresholded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()