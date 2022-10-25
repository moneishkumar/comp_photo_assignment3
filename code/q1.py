import pdb
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy import signal
import copy



def main():
    pass

def gkern(kernlen=24, std=0.01):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def piecewise_bilateral(image, image2, gamma_s, gamma_r):


    """
    gamma_r = [0.05, 0.25]
    gamma_s = 
    """
    # gamma_r : 0.05 to 0.25


    # Normalize the image to [0,1]
    image = image/255
    image2 = image2/255

    
    lam = 0.01
    minI = np.min(image2) - lam
    maxI = np.max(image2) + lam
    
    NB_SEGMENTS = int(np.ceil((maxI - minI)/gamma_r))

    h, w, c = image.shape
    bgr = []

    print(np.max(image), np.max(image2), NB_SEGMENTS)
    for channel in range(3):
        
        segments = []
        for i in range(0, NB_SEGMENTS):
            i_j = minI + i*((maxI - minI)/NB_SEGMENTS)
            G_j = np.exp(- ((image2[:,:,channel] - i_j)**2)/(2*(gamma_r**2)))
            K_j = cv2.GaussianBlur(G_j, ksize = (0,0),sigmaX=gamma_s)
            H_j = G_j*image[:,:,channel]
            H_st_j = cv2.GaussianBlur(H_j, ksize = (0,0), sigmaX=gamma_s)
            J_j = H_st_j/K_j
            #pdb.set_trace() 
            segments.append(J_j)

        segments = np.array(segments)
        points = (np.linspace(0,segments.shape[0], segments.shape[0]), \
                np.linspace(0,segments.shape[1],segments.shape[1]),\
                np.linspace(0,segments.shape[2],segments.shape[2]))

        vals = [(image[i,j,channel], i, j) for i in range(image.shape[0]) for j in range(image.shape[1])]
        inter_vals = interpn(points, segments,vals)
        bgr.append(np.reshape(inter_vals, (h,w)))
    
    bgr_bi = np.dstack(bgr)


    return bgr_bi

def join_bilateral(img1, img2, gamma_s, gamma_r, kernel_size):
    '''
    
    kernel_size: goes from 24 to 48
    '''
    # Normalize image
    img1 = img1/255
    img2 = img2/255

    h, w, c = img1.shape

    im_NR = np.zeros(img1.shape)
    img1 = cv2.copyMakeBorder(img1, kernel_size, kernel_size, kernel_size, kernel_size, cv2.BORDER_CONSTANT, (0,0,0))
    img2 = cv2.copyMakeBorder(img2, kernel_size, kernel_size, kernel_size, kernel_size, cv2.BORDER_CONSTANT, (0,0,0))

    #
    g_kernel = gkern(kernlen=2*kernel_size+1, std=gamma_s)

    st_g_kernel = np.dstack([g_kernel, g_kernel, g_kernel])
    for channel in range(c):
        for i in range(kernel_size, h):
            print(i)
            for j in range(kernel_size, w):
                g_r = img2[i-kernel_size:i+kernel_size+1,j-kernel_size:j+kernel_size+1,channel] - img2[i,j,channel]
                g_r = np.exp(-(g_r**2)/(2*(gamma_r**2)))
                k = np.sum(g_kernel*g_r)
                # pdb.set_trace()
                im_NR[i-kernel_size,j-kernel_size,channel] = (1/k)*np.sum(g_kernel*g_r*img1[i-kernel_size:i+kernel_size+1,j-kernel_size:j+kernel_size+1,channel])
                
        plt.imshow(im_NR[:,:,channel])
        plt.show()
        pass

def gamma_correction(img):

    val = 0.0404482

    idxs = img <= val
    img[idxs] = img[idxs]/12.92

    idxs = img > val
    img[idxs] = ((img[idxs]+0.055)/1.055)**(2.4)

    return img

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='CompPhoto HW3')
    parser.add_argument('--method',default='PiecewiseBilateral', help='Method that needs to be done')
    

    parser.add_argument('--image1',type=str, default = "data/lamp/lamp_ambient.tif", help='Intensity Influence')
    parser.add_argument('--image2',type=str, default = "data/lamp/lamp_flash.tif", help='Intensity Influence')

    ### For piecewise_bilateral
    parser.add_argument('--gamma_r',type=float, default = 0.05, help='Intensity Influence')
    parser.add_argument('--gamma_s', type=float, default = 0.15, help='Spatial Kernel')
    parser.add_argument('--kernel', type=int, default = 12, help='Kernel size')


    args = parser.parse_args()
    
    if args.method == "PiecewiseBilateral":

        image_path = "data/lamp/lamp_ambient.tif"
        img = cv2.imread(image_path)

        bgr_bi = piecewise_bilateral(img, img, args.gamma_s, args.gamma_r)

        cv2.imwrite(f'./output/{args.method}.png', ((bgr_bi/np.max(bgr_bi))*255).astype(np.uint8))
        
    elif args.method == "JointBilateral":

        image_path1 = "data/lamp/lamp_ambient.tif"
        A = cv2.imread(image_path1)

        image_path2 = "data/lamp/lamp_flash.tif"
        F = cv2.imread(image_path2)

        bgr_bi = piecewise_bilateral(A, F, args.gamma_s, args.gamma_r)
        
        cv2.imwrite(f'./output/{args.method}.png', ((bgr_bi/np.max(bgr_bi))*255).astype(np.uint8))

        #join_bilateral(img1, img2, args.gamma_s, args.gamma_r, args.kernel)
    
    elif args.method == "DetailTransfer":

        image_path1 = "data/lamp/lamp_ambient.tif"
        image_path1 = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/exposure4.jpg"
        A = cv2.imread(image_path1)

        image_path2 = "data/lamp/lamp_flash.tif"
        image_path2 = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/exposure3.jpg"
        F = cv2.imread(image_path2)

        #pdb.set_trace()
        A_NR = piecewise_bilateral(A, F, args.gamma_s, args.gamma_r)
        F_base = piecewise_bilateral(F, F, args.gamma_s, args.gamma_r)
        
        epsi = 1e-3

        weights = (((F/255)+epsi)/(F_base+epsi))
        weights = np.clip(weights, a_min =0, a_max =1)
        A_D = A_NR*weights
        
        cv2.imwrite(f'./output/{args.method}_my.jpg', ((A_D/np.max(A_D))*255).astype(np.uint8))
    
    elif args.method == "SSMasking":
        
        image_path1 = "data/lamp/lamp_ambient.tif"
        image_path1 = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/exposure4.jpg"
        
        A = cv2.imread(image_path1)
        A_ISO = 1600

        image_path2 = "data/lamp/lamp_flash.tif"
        image_path2 = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/exposure1.jpg"
        
        F = cv2.imread(image_path2)
        F_ISO = 200

        shadow_thresh = 0.1

        A_gc = gamma_correction(A.copy())
        F_lin = gamma_correction(F.copy())
        
        A_lin = A_gc*F_ISO/A_ISO

        mask = (F_lin-A_lin) < shadow_thresh
        
        # plt.imshow(mask)
        # pdb.set_trace()

        A_NR = piecewise_bilateral(A, F, args.gamma_s, args.gamma_r)
        A_base = piecewise_bilateral(A, A, args.gamma_s, args.gamma_r)
        F_base = piecewise_bilateral(F, F, args.gamma_s, args.gamma_r)
        
        epsi = 1e-3
        weights = (((F/255)+epsi)/(F_base+epsi))
        weights = np.clip(weights, a_min =0, a_max =1)
        A_D = A_NR*weights

        pdb.set_trace()
        A_F = (1-mask)*A_D + mask*A_base
        
        cv2.imwrite(f'./output/{args.method}_my.jpg', ((A_D/np.max(A_D))*255).astype(np.uint8))
        pass

    #main()

'''
python q1.py --method JointBilateral
'''