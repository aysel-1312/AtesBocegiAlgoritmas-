import numpy as np
from random import random
import numpy.matlib

#parametreler tanımlandı
iter = 15  #iterasyon sayısı
LB = -10 #Alt sınır
UB = 10 #üst sınır
D = 2  #Boyut
N = 100  #ateş böceği sayısı
gamma = 1 #Işık Soğurma Katsayısı(sabit emilim katsayısı) 0.01 ile 100 arasında değer alır
alpha = 0.2  #raslantı değişkeni
Beta0 = 2   #ateşböceğinin çekiciliği
teta = 0.97 #Rastgelelik azaltma faktörü 

#Fonksiyonlar yazıldı
def sphere(xx):  #sphere fonksiyonu tanımlandı

    d = xx.shape[2 - 1]
    sum = []
    for k in range(0, xx.shape[1 - 1]):

        for ii in range(0, d):

            xi = xx[k][ii]

        sum.append(xi**2)

    return sum


def griewank(xx): #griewank fonksiyonu tanımlandı

    nligne, ncol = np.shape(xx)
    ouput = []
    for k in range(0, nligne):

        xi = xx[k]

        formul = (np.sum(xi**2) / 4000) - \
            np.prod(np.cos(xi / np.sqrt(range(1, ncol + 1)))) + 1
        ouput.append(formul)

    return ouput


#func = 'griewank' #Fonsiyonları çağırma
func ='sphere'

gbest = float('inf') #Şimdiye Kadar Bulunan En İyi Çözüm  Sonsuz (infinity). Sonsuz büyük değere sahip bir değişkeni ayarlamak için kullanılır. Yani, değeri sonsuz olarak ayarlar .

fly = np.empty([N, D + 1])  #elemanların tamamı rastgele atanan dizi oluşturur(NxD+1 boyutunda) girişleri başlatmadan fly boş bir dizi.

# print fly

cost = []  #maliyet için dizi tanımlıyoruz

gbest_position = []  #şimdiye kadar bulunan en iyi çözümün konum dizisi tanımlıyoruz



# popülasyon oluşturma(N ateşböceğinin ilk konumlarının oluşturulması)
for i in range(0, N):

    pos = np.random.uniform(0, 1, (1, D)) * (UB - LB) #rasgele konum belirledik.
    # print pos
    cost = eval('%s(pos)' % (func))#maliyeti fonsiyon içinde değerlendiriyoruz.

    fly[i] = pos.tolist()[0] + cost #her ateşböceği için maliyet ve pozisyon listeside üzerine eklenerek fly dizisine atandı


    if cost < gbest: #her ateşböceği maliyeti en iyi çözüm ile karşılaştırıldı eğer maliyet hesabı en iyi çözümden küçükse

        gbest = cost  #en iyi maliyet en iyi çözüme atandı
        gbest_position.append(pos) # en iyi çözümün pozisyonuna ekle konumu da

#Ateşböceği algoritması 
def cozum(fly, N, iter, dim, gbest, alpha):
   
    t = 0
    while t < iter: #t iterasyon sayısından küçük oldukça devam edecek bir döngü
        #ateşböceği sayısı kadar tekrarlanacak Nx1 boyutunda tekrarlanır.
        gecici_fly = numpy.matlib.repmat(fly, N, 1)#repmat dizinin kopyalarını tekrarlar

        for i in range(0, N):#0 dan N kadar i sayısı döndürülecek

            gecici_fly[i][dim] = float('inf') #geçici ateşböceği sonsuz değerlerle başlatılır.her işlemden sonra değiştirilmesi için.

            yogunluk_i = fly[i][dim] #ve bu değer i nin yoğunluğuna atanır.

            for j in range(0, N):#0 dan N kadar j sayısı döndürülecek

                yogunluk_j = fly[j][dim]#aynı şekilde j ninde yoğunluğuna atanır.

                if yogunluk_i < yogunluk_j: #bu yoğunluklar karşılaştırılır. 

                    xi = fly[i][:dim-1 ] #yoğunluk sonucuna göre xi ve xj değerlerine atanır.
                    xj = fly[j][:dim-1 ]

                    epsilon = np.random.normal(0.0, (UB - LB) / 12, (1, dim))#epsilon değerinin hesaplanması.
                    r = np.sqrt(np.sum((xi - xj)**2)) #iki ateşböceği arasındaki uzaklık (i ve j)
                    beta = Beta0 * np.exp(-gamma * r**2)#betanın hesaplanması
                   

                      #ateşböceğinin yeni yoğunluk hesaplanması
                    new_val = yogunluk_i + beta * \
                        (yogunluk_j - yogunluk_i) + alpha * epsilon

                    # print (new_val)
                    if np.max(new_val) > UB:#yeni değer  üst değerden büyükse yeni değer üst değere eşittir
                      iid = np.argmax(new_val)
                      new_val[i][iid] = UB
                    if np.min(new_val) < LB:#yeni değer alt değerden küçükse yeni değer alt değere eşittir
                      iid = np.argmin(new_val)
                      new_val[i][iid] = LB

                    cost = eval('%s(new_val)' % (func)) #maliyeti fonsiyon içinde değerlendiriyoruz.Güncelliyoruz.
                    # print cost

                    if cost[0] < gecici_fly[i][dim]:#eğer maliyet, ateşböceğinin i. indexnden küçükse

                        gecici_fly[i][:dim] = new_val #ateşböceği konumu  yeni değere eşit olur.Tüm sütunu yazdırılır.
                        gecici_fly[i][dim] = cost[0] #ateşböceği maliyeti de  aynı şekilde eşit olur. 

            if gecici_fly[i][dim] < gbest: #eğer geçici ateşböceği değeri en iyi çözümden küçükse 
                gbest_position = gecici_fly[i][:dim] #en iyi çözüm konumunu da atadı 
                gbest = gecici_fly[i][dim] #geçici ateşböceği değerini gbest değerine atadı

        t += 1 #iterasyon arttırma yapıldı
        alpha = alpha * teta # Nemli Mutasyon Katsayısı kat sayısı

        #sonuçları yazdırma
        print ('iterasyon :')
        print(t)
        print ('Cozum :')
        print(gbest)
        print ('Cozum_Konumu:')
        print(gbest_position)

cozum(fly, N, iter, D, gbest, alpha)