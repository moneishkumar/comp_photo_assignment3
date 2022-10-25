from curses import A_CHARTEXT
import pdb
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy import signal
import copy

from torch import det


def gradient(img):
    # Assuming that img is a scalar field i.e greyscale image
    I_y = np.diff(img, n = 1, axis = 0, append=0)
    I_x = np.diff(img, n = 1, axis = 1, append=0)
    
    return np.dstack([I_x, I_y])

def divergance(gradient):
    # Dot product with the nabla

    I_yy = np.diff(gradient[:,:,1], n = 1, axis = 0, prepend=0)
    I_xx = np.diff(gradient[:,:,0], n = 1, axis = 1, prepend=0) 

    return I_xx + I_yy
    

def laplacian(img):
    
    laplacian = np.array([ [0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])

    return signal.convolve2d(img, laplacian, mode = 'same', boundary= 'fill', fillvalue=0)


def GFI_with_CGD(D, I_init, B, I_bound, epsilon= 0.001, N = 1000):

    I_star = B*I_init + (1-B)*I_bound
    r = B*(D - laplacian(I_star))
    d = r    
    delta = np.sum(d**2)
    n = 0

    #pdb.set_trace()
    while (delta > epsilon and n < N):
        #print(n, delta)
        q = laplacian(d)
        neta = delta/(np.sum(d*q))
        I_star = I_star + B*(neta*d)
        r = B*(r - neta*q)
        delta_old = delta
        delta = np.sum(r**2)
        beta = delta/delta_old
        d = r + beta*d
        n = n + 1
    
    return I_star

def test(A):

    ### TESTING ###
    img = A[:, :, 0]#np.random.randint(2,size=(5, 5) )
    g = gradient(img)
    lap1 = divergance(g)
    lap2 = laplacian(img)

    i_zero = np.zeros(img.shape)
    B = np.zeros(img.shape)

    B[0,:] = 1
    B[-1,:] = 1
    B[:,0] = 1
    B[:,-1] = 1

    I_bound = B*img
    I_inti = GFI_with_CGD(lap2, i_zero, 1-B, I_bound)

    plt.imshow(I_inti)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='CompPhoto HW3')
    parser.add_argument('--method',default='PiecewiseBilateral', help='Method that needs to be done')
    

    # parser.add_argument('--image1',type=str, default = "data/museum/museum_ambient.png", help='Image Ambient')
    # parser.add_argument('--image2',type=str, default = "data/museum/museum_flash.png", help='Image flash')

    parser.add_argument('--image1',type=str, default = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/flash5.jpg", help='Image Ambient')
    parser.add_argument('--image2',type=str, default = "/home/moneish/CMU/fall_22/comp_photo/assgn3/gphoto/flash4.jpg", help='Image flash')

    args = parser.parse_args()

    
    A_c = cv2.imread(args.image1)
    phi_dash_c = cv2.imread(args.image2)

    A_c = cv2.resize(A_c, (A_c.shape[1]//8, A_c.shape[0]//8))
    phi_dash_c = cv2.resize(phi_dash_c, (phi_dash_c.shape[1]//8, phi_dash_c.shape[0]//8))
    #test(A_c)

    channels = []
    for i in range(3):
        print(i)
        A = A_c[:,:,i]/255
        phi_dash = phi_dash_c[:,:,i]/255
        
        DA = gradient(A.copy())
        Dphi_dash = gradient(phi_dash.copy())

        # plt.subplot(3,2,1)
        # plt.imshow(DA[:,:,0])

        # plt.subplot(3,2,2)
        # plt.imshow(DA[:,:,1])

        # plt.subplot(3,2,3)
        # plt.imshow(Dphi_dash[:,:,0])

        # plt.subplot(3,2,4)
        # plt.imshow(Dphi_dash[:,:,1])

        M = np.abs(Dphi_dash[:,:,0]*DA[:,:,0] + Dphi_dash[:,:,1]*DA[:,:,1])

        # temp = M >= 0.005

        den = (np.sqrt(Dphi_dash[:,:,0]**2 + Dphi_dash[:,:,1]**2)) * (np.sqrt(DA[:,:,0]**2 + DA[:,:,1]**2))
        M = M/ (den+0.00001)

        #pdb.set_trace()

        sigma = 10 #40 
        tau = 0.6 # 0.9 

        w_s = np.tanh(sigma*(phi_dash- tau))

        #pdb.set_trace()
        w_s = (w_s - np.min(w_s))
        w_s = w_s/np.max(w_s)

        phi_star_x = w_s*DA[:,:,0] + (1-w_s)*(M*Dphi_dash[:,:,0] + (1-M)*DA[:,:,0])
        phi_star_y = w_s*DA[:,:,1] + (1-w_s)*(M*Dphi_dash[:,:,1] + (1-M)*DA[:,:,1])


        # plt.subplot(3,2,5)
        # plt.imshow(phi_star_x)

        # plt.subplot(3,2,6)
        # plt.imshow(phi_star_y)

        phi_star = np.dstack([phi_star_x, phi_star_y])

        B = np.zeros(A.shape)
        B[0,:] = 1
        B[-1,:] = 1
        B[:,0] = 1
        B[:,-1] = 1

        #plt.show()

        i_zero = np.zeros(A.shape)
        I_bound = B*phi_dash
        I_inti = GFI_with_CGD(divergance(phi_star), i_zero, 1-B, I_bound)

        channels.append(I_inti.copy())
        #pdb.set_trace()


    bgr = np.dstack(channels)
    bgr = np.clip(bgr, a_min=0, a_max=1)
    

    cv2.imwrite(f'./data/museum/GDP_my.png', (bgr*255).astype(np.uint8))


