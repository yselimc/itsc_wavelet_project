import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt #wavelet için

paths=["digital_checkered.png","satellite_image.jpg","normal.jpg","pattern.jpg"] #kullanılacak resim
TRESHOLDS=[0.05, 0.10, 0.15, 0.20, 0.25,0.30,0.35]    #sıkıştırma miktarı ve kaliteyi belirleyen eşik değeri
WAVELET_NAMES= ["haar","db2","sym4"] #kullanılacak WAVELET'in türü

#resmi grayscale yapıp float matris haline getiren metot
def prep_img(p):
    img=mpimg.imread(p)
    #print(img.shape)
    
    if img.ndim==3:#ndim 3se resim renklidir, dot productla gri yaparız
        gray=np.dot(img[...,:3],[0.2989,0.5870,0.1140])
    else:#ndim 3 değilse 2 ise resim zaten gridir. birşey yapmaya gerek yok
        gray=img
        
    gray=gray.astype(np.float32)
    
    if gray.max()>1.5:#katsayılar 1 ile 0 arasında olacak işlem kolaylığı için
        gray=gray/255.0
    
    #print(img.shape,img.ndim)
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
    #threshold uygulanmış matrisi ve thresholda takılmış eleman sayısını döndürür
    return copy,smalls_count

#sıkıştırılmış resmi eski haline dönüştüren metot
def reconstruct(coeffs2_thr,gray,WAVELET_NAME):
    rec = pywt.idwt2(coeffs2_thr, WAVELET_NAME)#ters wavelet
    #print("reconstructed shape:", rec.shape)
    min_h = min(gray.shape[0], rec.shape[0])
    min_w = min(gray.shape[1], rec.shape[1])

    gray_c= gray[:min_h,:min_w]
    reconstructed_c= rec[:min_h,:min_w]
    return gray_c,reconstructed_c #orijinal ve geri oluşturulmuş matrisi döndürür

#geri dönüştürülmüş resim ile ilk resim arasındaki farkı hesaplayan metod
def calculate_mse_psnr(gray_c, reconstructed_c):
    diff= gray_c - reconstructed_c #fark matrisi
    mse= np.mean(diff**2)# mse hesapladık (psnr formulünde kullanabilmek için lazımdı)
    if mse== 0:
        psnr= float( 'inf')
    else:
        max_val= 1.0
        psnr= 10*np.log10((max_val**2)/mse) #PSNR için genel formul bu
    return mse,psnr

def compress_and_evaluate_images(path,wavelet_name,TRESHOLD):
    #resmi hazırla
    gray=prep_img(path)

    #wavelet uygula
    coeffs2=perform_wavelet(gray,wavelet_name)
    coeff_A,(coeff_H,coeff_V,coeff_D)=coeffs2

    #treshold uygula
    coeff_A_thr, num_of_zeroed_coeffA= apply_threshold(coeff_A, TRESHOLD)
    coeff_H_thr, num_of_zeroed_coeffH= apply_threshold(coeff_H, TRESHOLD)
    coeff_V_thr, num_of_zeroed_coeffV= apply_threshold(coeff_V, TRESHOLD)
    coeff_D_thr, num_of_zeroed_coeffD= apply_threshold(coeff_D, TRESHOLD)

    #resmi geri oluştur
    coeffs2_thr = (coeff_A_thr, (coeff_H_thr, coeff_V_thr, coeff_D_thr))
    gray_c,reconstructed_c=reconstruct(coeffs2_thr,gray,wavelet_name)

    #farkı ölç
    mse, psnr= calculate_mse_psnr(gray_c, reconstructed_c)
    #tablolarda kullanmak için sparsity yi de hesaplıyoruz ve psnr ile birlikte döndürüyoruz
    total= coeff_A.size+ coeff_H.size+ coeff_V.size+ coeff_D.size
    zeroed= num_of_zeroed_coeffA+ num_of_zeroed_coeffH+ num_of_zeroed_coeffV+ num_of_zeroed_coeffD
    sparsity= zeroed/total
    return psnr,sparsity

#projedeki tüm fonksiyonları kullanan ve grafikleri yapan fonksiyon
def print_tradeoff_tables(path,wavelet_names,thresholds):
    plt.figure(figsize=(8,5))
    
    for wavelet in wavelet_names:
        xs=[] #thresholdlar icin x ekseni
        ys=[] #thresholdlar icin y ekseni
        for threshold in thresholds:
            psnr,sparsity =compress_and_evaluate_images(path,wavelet,threshold)
            xs.append(sparsity*100)
            ys.append(psnr)
            
        plt.plot(xs,ys,marker='o',label=wavelet)
        for x,y,t in zip(xs,ys,thresholds):
            plt.text(x,y,f"{t}",fontsize=8)
    
    plt.xlabel("sparsity(%)") #(kabaca sıkıştırma miktarı)
    plt.ylabel("psnr (dB) (quality)") #(sıkıştırılıp geri dönüştürülen resmin kalitesi)
    plt.title(f"trade-off (for {path})")
    plt.grid(True,alpha=0.3)
    plt.axhline(30, color="yellow", linestyle="-", linewidth=3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#approximation, horizontal, diagonal ve vertical subbandları görselleştirmek için. gereken metot
def show_subbands_of_wavelet(path,wavelet_name):
    gray=prep_img(path)
    coeffs2=pywt.dwt2(gray,wavelet_name)
    LL,(LH,HL,HH)=coeffs2
    
    plt.figure(figsize=(8,8))
    
    plt.subplot(2,2,1)
    plt.imshow(LL,cmap='gray')
    plt.title( "LL (Approximation)")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(LH,cmap='gray')
    plt.title( "LH (Horizontal details)")
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(HL,cmap='gray')
    plt.title( "HL (Vertical details)")
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(HH,cmap='gray')
    plt.title( "HH (Diagonal details)")
    plt.axis('off')
    
    plt.suptitle("subbands in wavelet",fontsize=14)
    plt.tight_layout()
    plt.show()
    

#main
for path in paths:
    print_tradeoff_tables(path,WAVELET_NAMES,TRESHOLDS)

#horizontal, vertical, diagonal subbandları görselleştirmek için.
show_subbands_of_wavelet("aybu.jpg",WAVELET_NAMES[0])