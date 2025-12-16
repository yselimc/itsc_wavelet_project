import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt #wavelet için

path="sozen.png" #kullanılacak resim
TRESHOLD=0.3    #sıkıştırma miktarı ve kaliteyi belirleyen eşik değeri
WAVELET_NAME="haar" #kullanılacak WAVELET'in türü

#resmi grayscale yapıp float matris haline getiren metot
def prep_img(p):
    img=mpimg.imread(p)
    print(img.shape)
    
    if img.ndim==3:#ndim 3se resim renklidir, dot productla gri yaparız
        gray=np.dot(img[...,:3],[0.2989,0.5870,0.1140])
        pass
    else:#ndim 3 değilse 2 ise resim zaten gridir. birşey yapmaya gerek yok
        gray=img
        
    gray=gray.astype(np.float32)
    
    if gray.max()>1.5:#katsayılar 1 ile 0 arasında olacak işlem kolaylığı için
        gray=gray/255.0
    
    print(img.shape,img.ndim)
    return gray

#waveleti uygulayan metot
def perform_wavelet(gray,wavelet_name):
    c2=pywt.dwt2(gray,wavelet_name)
    return c2


#resmi hazırla
gray=prep_img(path)

#wavelet uygula
coeffs2=perform_wavelet(gray,WAVELET_NAME)
coeff_A,(coeff_H,coeff_V,coeff_D)=coeffs2