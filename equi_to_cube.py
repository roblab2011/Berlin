# modified https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map

from math import pi, atan2, hypot, floor
import numpy as np
from numba import jit

@jit(nopython=True)
def clip(value, min, max):
    if value < max and value > min:
        return value
    elif value > max:
        return max
    else:
        return min

# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
@jit(nopython=True)
def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# convert using an inverse transformation
@jit(nopython=True)
def convertBack(inPix,outPix):
    inSize = inPix.shape
    outSize = outPix.shape
    edge = int(inSize[1]/4)   # the length of each edge in pixels
    for i in range(outSize[1]):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        if face==2:
            rng = range(0,edge*3)
        else:
            rng = range(edge,edge*2)

        for j in rng:
            if j<edge:
                face2 = 4 # top
            elif j>=2*edge:
                face2 = 5 # bottom
            else:
                face2 = face

            (x,y,z) = outImgToXYZ(i,j,face2,edge)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + pi)/pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[clip(vi,0,inSize[0]-1),ui % inSize[1]]
            B = inPix[clip(vi,0,inSize[0]-1),u2 % inSize[1]]
            C = inPix[clip(v2,0,inSize[0]-1),ui % inSize[1]]
            D = inPix[clip(v2,0,inSize[0]-1),u2 % inSize[1]]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[j,i] = (int(round(r)),int(round(g)),int(round(b)))

def equi_to_cube(image):
    imgIn = image
    inSize = imgIn.shape
    imgOut = np.zeros((int(inSize[1]*3/4),inSize[1],3), dtype=np.uint8)
    convertBack(imgIn, imgOut)
    return imgOut