clc()
import numpy as np
import cv2
import scipy.signal as signal

# lowpass filter
b, a = signal.butter(8,0.034,'lowpass')

# read image of projecting fringe on base plane
Bfringe01=(cv2.imread('Base1.jpg',cv2.IMREAD_GRAYSCALE))
Bfringe02=(cv2.imread('Base2.jpg',cv2.IMREAD_GRAYSCALE))
Bfringe03=(cv2.imread('Base3.jpg',cv2.IMREAD_GRAYSCALE))
Bfringe04=(cv2.imread('Base4.jpg',cv2.IMREAD_GRAYSCALE))

# read imae of projecting fringe on correction plane
Kfringe01=(cv2.imread('Correction1.jpg',cv2.IMREAD_GRAYSCALE))
Kfringe02=(cv2.imread('Correction2.jpg',cv2.IMREAD_GRAYSCALE))
Kfringe03=(cv2.imread('Correction3.jpg',cv2.IMREAD_GRAYSCALE))
Kfringe04=(cv2.imread('Correction4.jpg',cv2.IMREAD_GRAYSCALE))

#read imae of projecting fringe on the coin
fringe01=(cv2.imread('Coin1.jpg',cv2.IMREAD_GRAYSCALE))
fringe02=(cv2.imread('Coin2.jpg',cv2.IMREAD_GRAYSCALE))
fringe03=(cv2.imread('Coin3.jpg',cv2.IMREAD_GRAYSCALE))
fringe04=(cv2.imread('Coin4.jpg',cv2.IMREAD_GRAYSCALE))

#using median filter and low pass filter on images
Bfringe_medianBlur1=cv2.medianBlur(Bfringe01,9)
Bfringe_medianBlur2=cv2.medianBlur(Bfringe02,9)
Bfringe_medianBlur3=cv2.medianBlur(Bfringe03,9)
Bfringe_medianBlur4=cv2.medianBlur(Bfringe04,9)
Bfringe1 = signal.filtfilt(b,a,Bfringe_medianBlur1)
Bfringe2 = signal.filtfilt(b,a,Bfringe_medianBlur2)
Bfringe3 = signal.filtfilt(b,a,Bfringe_medianBlur3)
Bfringe4 = signal.filtfilt(b,a,Bfringe_medianBlur4)
Bfringe_EH1 = ((Bfringe1)-127.5)/127.5
Bfringe_EH2 = ((Bfringe2)-127.5)/127.5
Bfringe_EH3 = ((Bfringe3)-127.5)/127.5
Bfringe_EH4 = ((Bfringe4)-127.5)/127.5      #expressing intensity in number between -1~+1

Kfringe_medianBlur1=cv2.medianBlur(Kfringe01,9)
Kfringe_medianBlur2=cv2.medianBlur(Kfringe02,9)
Kfringe_medianBlur3=cv2.medianBlur(Kfringe03,9)
Kfringe_medianBlur4=cv2.medianBlur(Kfringe04,9)
Kfringe1 = signal.filtfilt(b,a,Kfringe_medianBlur1)
Kfringe2 = signal.filtfilt(b,a,Kfringe_medianBlur2)
Kfringe3 = signal.filtfilt(b,a,Kfringe_medianBlur3)
Kfringe4 = signal.filtfilt(b,a,Kfringe_medianBlur4)
Kfringe_EH1 = ((Kfringe1)-127.5)/127.5
Kfringe_EH2 = ((Kfringe2)-127.5)/127.5
Kfringe_EH3 = ((Kfringe3)-127.5)/127.5
Kfringe_EH4 = ((Kfringe4)-127.5)/127.5      #expressing intensity in number between -1~+1

fringe_medianBlur1=cv2.medianBlur(fringe01,9)
fringe_medianBlur2=cv2.medianBlur(fringe02,9)
fringe_medianBlur3=cv2.medianBlur(fringe03,9)
fringe_medianBlur4=cv2.medianBlur(fringe04,9)
fringe1 = signal.filtfilt(b,a,fringe_medianBlur1)
fringe2 = signal.filtfilt(b,a,fringe_medianBlur2)
fringe3 = signal.filtfilt(b,a,fringe_medianBlur3)
fringe4 = signal.filtfilt(b,a,fringe_medianBlur4)
fringe_EH1 = ((fringe1)-127.5)/127.5
fringe_EH2 = ((fringe2)-127.5)/127.5
fringe_EH3 = ((fringe3)-127.5)/127.5
fringe_EH4 = ((fringe4)-127.5)/127.5      #expressing intensity in number between -1~+1

#choose the range of image(only include the coin)to shorten the processing time
vert_low = 800
vert_high = 2150
hori_low = 500
hori_high = 1900

vert_range = vert_high - vert_low
hori_range = hori_high - hori_low

BI1 = np.zeros((vert_range,hori_range))
BI2 = np.zeros((vert_range,hori_range))
BI3 = np.zeros((vert_range,hori_range))
BI4 = np.zeros((vert_range,hori_range))
Bphie = np.zeros((vert_range,hori_range))
Bunphie = np.zeros((vert_range,hori_range))
KI1 = np.zeros((vert_range,hori_range))
KI2 = np.zeros((vert_range,hori_range))
KI3 = np.zeros((vert_range,hori_range))
KI4 = np.zeros((vert_range,hori_range))
Kphie = np.zeros((vert_range,hori_range))
Kunphie = np.zeros((vert_range,hori_range))
I1 = np.zeros((vert_range,hori_range))
I2 = np.zeros((vert_range,hori_range))
I3 = np.zeros((vert_range,hori_range))
I4 = np.zeros((vert_range,hori_range))
phie = np.zeros((vert_range,hori_range))
unphie = np.zeros((vert_range,hori_range))

#use four step phase shifting technique to obtain the origin pahse of each image
for i in range(vert_low,vert_high):
    for j in range(hori_low,hori_high):
        I1[i-vert_low,j-hori_low] = fringe_EH1[i,j]
        I2[i-vert_low,j-hori_low] = fringe_EH2[i,j]
        I3[i-vert_low,j-hori_low] = fringe_EH3[i,j]
        I4[i-vert_low,j-hori_low] = fringe_EH4[i,j]
        phie[i-vert_low,j-hori_low] = np.arctan((I4[i-vert_low,j-hori_low]-I2[i-vert_low,j-hori_low])/(I1[i-vert_low,j-hori_low]-I3[i-vert_low,j-hori_low]))
        KI1[i-vert_low,j-hori_low] = Kfringe_EH1[i,j]
        KI2[i-vert_low,j-hori_low] = Kfringe_EH2[i,j]
        KI3[i-vert_low,j-hori_low] = Kfringe_EH3[i,j]
        KI4[i-vert_low,j-hori_low] = Kfringe_EH4[i,j]
        Kphie[i-vert_low,j-hori_low] = np.arctan((KI4[i-vert_low,j-hori_low]-KI2[i-vert_low,j-hori_low])/(KI1[i-vert_low,j-hori_low]-KI3[i-vert_low,j-hori_low]))
        BI1[i-vert_low,j-hori_low] = Bfringe_EH1[i,j]
        BI2[i-vert_low,j-hori_low] = Bfringe_EH2[i,j]
        BI3[i-vert_low,j-hori_low] = Bfringe_EH3[i,j]
        BI4[i-vert_low,j-hori_low] = Bfringe_EH4[i,j]
        Bphie[i-vert_low,j-hori_low] = np.arctan((BI4[i-vert_low,j-hori_low]-BI2[i-vert_low,j-hori_low])/(BI1[i-vert_low,j-hori_low]-BI3[i-vert_low,j-hori_low]))



unphie[:,0]=phie[:,0]
Kunphie[:,0]=Kphie[:,0]
Bunphie[:,0]=Bphie[:,0]

#phase unwrapping by compensating N*pi
for s1 in range(vert_low,vert_high):
    N=0
    check = -0.9
    KN=0
    Kcheck = -2.95
    BN = 0
    Bcheck = -2.95
    for s2 in range(hori_low,hori_high):
        unphie[s1-vert_low,s2-hori_low]=phie[s1-vert_low,s2-hori_low]
        Kunphie[s1-vert_low,s2-hori_low]=Kphie[s1-vert_low,s2-hori_low]
        Bunphie[s1-vert_low,s2-hori_low]=Bphie[s1-vert_low,s2-hori_low]
        if unphie[s1-vert_low,s2-hori_low]-unphie[s1-vert_low,s2-hori_low-1]<check:
            N = N+1
            check = check-np.pi
        unphie[s1-vert_low,s2-hori_low] = unphie[s1-vert_low,s2-hori_low]+N*np.pi
        if Kunphie[s1-vert_low,s2-hori_low]-Kunphie[s1-vert_low,s2-hori_low-1]<Kcheck:
            KN = KN+1
            Kcheck = Kcheck-np.pi
        Kunphie[s1-vert_low,s2-hori_low] = Kunphie[s1-vert_low,s2-hori_low]+KN*np.pi
        if Bunphie[s1-vert_low,s2-hori_low]-Bunphie[s1-vert_low,s2-hori_low-1]<Bcheck:
            BN = BN+1
            Bcheck = Bcheck-np.pi
        Bunphie[s1-vert_low,s2-hori_low] = Bunphie[s1-vert_low,s2-hori_low]+BN*np.pi


H1 = np.zeros((vert_range,hori_range))
H = np.zeros((vert_range,hori_range))
K = np.zeros((vert_range,hori_range))
H12 = np.zeros((vert_range,hori_range))
H22 = np.zeros((vert_range,hori_range))
K2 = np.zeros((vert_range,hori_range))

#rebuild the origin suface of the coin by using H = K*phase difference between the coin and base plane
for h1 in range(0,vert_range):
    for h2 in range (0,hori_range):
        H1[h1,h2] = ((Kunphie[h1,h2]-Bunphie[h1,h2]))       #obtain the constant K
        K[h1,h2] = 1/H1[h1,h2]
        H[h1,h2] = ((unphie[h1,h2]-Bunphie[h1,h2]))
        H[h1,h2] = K[h1,h2]*H[h1,h2]        #H = K*phase difference between the coin and base plane
