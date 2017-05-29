##### https://github.com/pabsaura

import cv2
# Importing the Opencv Library
import numpy as np
import numpy.fft as fft
from transform import four_point_transform
from scipy               import ndimage

def rgb2gray(x):
    return cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)

def orientation(x):
    return cv2.contourArea(x.astype(np.float32),oriented=True) >= 0

def fixOrientation(x):
    if orientation(x):
        return x
    else:
        return np.flipud(x)

def extractContours(g, minlen=50, holes=False):
    if holes:
        mode = cv2.RETR_CCOMP
    else:
        mode = cv2.RETR_EXTERNAL
    gt = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,-10)
    a,contours,b = cv2.findContours(gt.copy(), mode ,cv2.CHAIN_APPROX_NONE)
    ok = [fixOrientation(c.reshape(len(c),2)) for c in contours if cv2.arcLength(c,closed=True) >= minlen]
    return ok

def invar(c, wmax=10):
    z = c[:,0]+c[:,1]*1j
    f  = fft.fft(z)
    fa = abs(f)

    s = fa[1] + fa[-1]
    fp = fa[2:wmax+2];
    fn = np.flipud(fa)[1:wmax+1];
    return np.hstack([fp, fn]) / s

def mindist(c,mods,labs):
    import numpy.linalg as la
    ds = [(la.norm(c-mods[m]),labs[m]) for m in range(len(mods)) ]
    return sorted(ds, key=lambda x: x[0])


img = cv2.imread("matriculas.jpg")

fromCenter = False
r = cv2.selectROI(img, fromCenter)

imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# Display cropped image
cv2.imshow("CropedImage", imCrop)

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(imCrop,cv2.COLOR_RGB2GRAY)
cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Gray Converted Image",img_gray)
#cv2.waitKey() # Wait for a keystroke from the user
# Display Image

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Noise Removed Image",noise_removal)
#cv2.waitKey() # Wait for a keystroke from the user
# Display Image
# Thresholding the image
ret,thresh_image = cv2.threshold(noise_removal,0,255,cv2.THRESH_OTSU)
cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Image after Thresholding",thresh_image)
#cv2.waitKey() # Wait for a keystroke from the user
# Display Image

# Applying Canny Edge detection
canny_image = cv2.Canny(thresh_image,250,255)
cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Image after applying Canny",canny_image)
#cv2.waitKey() # Wait for a keystroke from the user

canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
kernel = np.ones((3,3), np.uint8)
# Creating the kernel for dilation
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Dilation", dilated_image)



# Finding Contours in the image based on edges
new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
# Sort the contours based on area ,so that the number plate will be in top 10 contours
screenCnt = None
# loop over our contours
for c in contours:
 # approximate the contour
 peri = cv2.arcLength(c, True)
 approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error

 if len(approx) == 4:
  screenCnt = approx
  break

final = cv2.drawContours(imCrop, [screenCnt], -1, (0, 255, 0), 3)
# Drawing the selected contour on the original image
cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Image with Selected Contour",final)
#cv2.waitKey() # Wait for a keystroke from the user

# Masking the part other than the number plate
mask = np.zeros(img_gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(imCrop,imCrop,mask=mask)
warped = four_point_transform(new_image, screenCnt.reshape(4, 2))
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",warped)

###########################################################################
#
#                   HASTA AQUI LA DETECCION DE LA MATRICULA
#
###########################################################################

###########################################################################
#
#                  AHORA, RECONOCIMIENTO DE LA MATRICULA
#
###########################################################################
warped_gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
noise_removal = cv2.bilateralFilter(warped_gray,9,75,75)
ret,thresh_image = cv2.threshold(noise_removal,0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)


canny_image = cv2.convertScaleAbs(canny_image)
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)


things = sorted(extractContours(dilated_image), key=lambda x: x[0,0])

final = cv2.drawContours(np.zeros(warped.shape,np.uint8), things, -1, (0, 255, 0), 3)

# Drawing the selected contour on the original image
cv2.namedWindow("Contours in the plate",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Contours in the plate",final)

cv2.waitKey()
g = 255-rgb2gray(cv2.cvtColor( cv2.imread('shapes/platestemplates.jpg'), cv2.COLOR_BGR2RGB))
ret,thresh_image = cv2.threshold(g,0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
dilated_image2 = cv2.dilate(canny_image,kernel,iterations=1)


models = sorted(extractContours(dilated_image2), key=lambda x: x[0,0])

finalTemplate = cv2.drawContours(dilated_image2, models, -1, (255, 255, 0), 3)
#finalTemplate = cv2.drawContours(g, models, -1, (0, 255, 0), 3)

labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
feats = [invar(m) for m in models]
i=10

for x in things:
    font = cv2.FONT_HERSHEY_SIMPLEX
    d,l = mindist(invar(x),feats,labels)[0]
    i=i+40
    if d < 0.15:
        cx,cy = np.mean(x,0)
        print(l)
        cv2.putText(warped,l,(int(cx),int(cy)), font, 1,(255,0,255),2,cv2.LINE_AA)


cv2.destroyAllWindows()
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",warped)


cv2.waitKey()
