#####https://github.com/pabsaura
#Ejemplo de uso  python  Ej12V2.py -f pano/pano001.jpg -s pano/pano002.jpg -t  pano/pano003.jpg

import argparse
import cv2
import numpy as np

def rgb2gray(x):
    return cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
def t(h,x):
    return cv2.warpPerspective(x, desp((100,150)).dot(h),(1500,500))
def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

def unirImagenes(x,y):
    imageA = rgb2gray(x)
    imageB = rgb2gray(y)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(imageA, None)
    (kps2, descs2) = sift.detectAndCompute(imageB, None)
    cv2.imshow( "Keypoints",cv2.drawKeypoints(image=imageA,
                             outImage=None,
                             keypoints=kps,
                             flags=4, color = (128,0,0)) );

    cv2.waitKey(0)
    cv2.imshow( "Keypoints 2",cv2.drawKeypoints(image=imageB,
                             outImage=None,
                             keypoints=kps2,
                             flags=4, color = (128,0,0)) );
    cv2.waitKey(0)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descs2,descs,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)


    img3 = cv2.drawMatchesKnn(imageB,kps2,
                             imageA,kps,
                             [[m] for m in good],
                             flags=2,outImg=None,
                             matchColor=(128,0,0))
    cv2.imshow("Matches", img3)
    cv2.waitKey(0)

    src_pts = np.array([ kps [m.trainIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    dst_pts = np.array([ kps2[m.queryIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)

    matchesMask = mask.ravel()>0
    ok = [ good[k] for k in range(len(good)) if matchesMask[k] ]

    img4 = cv2.drawMatchesKnn(imageB,kps2,imageA,kps,[[m] for m in ok],flags=2,outImg=None,matchColor=(0,255,0))

    cv2.imshow("Matches with RANSAC", img4)
    cv2.waitKey(0)

    cv2.imshow("Grande" ,t(np.eye(3),imageB))
    cv2.waitKey(0)
    cv2.imshow("Grande 2",t(H,imageA))
    cv2.waitKey(0)
    union=np.maximum(t(np.eye(3),imageB), t(H,imageA))
    unionColor=np.maximum(t(np.eye(3),y), t(H,x))

    return unionColor

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
ap.add_argument("-t", "--third", required=True,
    help="path to the third image")
args = vars(ap.parse_args())


imageAC = cv2.imread(args["first"])
imageBC = cv2.imread(args["second"])
imageCC = cv2.imread(args["third"])

result = unirImagenes(imageAC,imageBC)

cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

resultFinal = unirImagenes(result,imageCC)
cv2.imshow("Result",resultFinal)

cv2.waitKey(0)
