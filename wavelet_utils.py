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

#wavelet uygulanmış matrise tresholdları uygulayan (sıkıştıran metot)
def apply_threshold(coeff_array,T):
    copy=coeff_array.copy()
    smalls=np.abs(copy)<T
    copy[smalls]=0
    smalls_count=smalls.sum()
    return copy,smalls_count

#sıkıştırılmış resmi eski haline dönüştüren metot
def reconstruct(coeffs2_thr,gray,WAVELET_NAME):
    rec = pywt.idwt2(coeffs2_thr, WAVELET_NAME)
    print("reconstructed shape:", rec.shape)
    min_h = min(gray.shape[0], rec.shape[0])
    min_w = min(gray.shape[1], rec.shape[1])

    gray_c = gray[:min_h, :min_w]
    reconstructed_c  = rec[:min_h, :min_w]
    return gray_c,reconstructed_c

#geri dönüştürülmüş resim ile ilk resim arasındaki farkı hesaplayan metod
def calculate_mse_psnr(gray_c, reconstructed_c):
    diff = gray_c - reconstructed_c
    mse = np.mean(diff**2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = 1.0
        psnr = 10 * np.log10((max_val**2)/mse) #PSNR için genel formul bu
    return mse,psnr

#resmi hazırla
gray=prep_img(path)

#wavelet uygula
coeffs2=perform_wavelet(gray,WAVELET_NAME)
coeff_A,(coeff_H,coeff_V,coeff_D)=coeffs2

#treshold uygula
coeff_A_thr, num_of_zeroed_coeffA  = apply_threshold(coeff_A, TRESHOLD)
coeff_H_thr, num_of_zeroed_coeffH  = apply_threshold(coeff_H, TRESHOLD)
coeff_V_thr, num_of_zeroed_coeffV  = apply_threshold(coeff_V, TRESHOLD)
coeff_D_thr, num_of_zeroed_coeffD  = apply_threshold(coeff_D, TRESHOLD)

#resmi geri oluştur
coeffs2_thr = (coeff_A_thr, (coeff_H_thr, coeff_V_thr, coeff_D_thr))
gray_c,reconstructed_c=reconstruct(coeffs2_thr,gray,WAVELET_NAME)

#farkı ölçmek için
mse, psnr= calculate_mse_psnr(gray_c, reconstructed_c)
print("MSE: ", mse)
print("PSNR (dB): ", psnr)