# Trendyol Hackathon 2025 8. Sıra Çözümü

Önemli not: Veri gizliliği kurallarından ötürü yarışma datası repoya dahil edilmemiştir. Kodlar büyük oranda kullanıma hazır haldedir, minör path/kütüphane ayarları gerekebilir.

Problem: Bu yarışmada katılımcılar, arama sonuçlarının tıklama ve satın almaya dönüşüm oranlarını artırmaya odaklanır. Hedef; belirli bir kullanıcı × arama terimi × tarih üçlüsü (bir oturum) için her ürünün olası aksiyonlarını (clicked / ordered) dikkate alıp, bu aksiyonların gerçekleşmesi en muhtemel ürünleri öne çıkaracak bir sıralama üretmektir.

Amaç: Her oturumda, ürünlerin tıklanma ve sipariş alma ihtimallerini gözeten bir puanlama ile listeleme kalitesini yükseltmek.

İş hedefi: Kullanıcıya daha alakalı ürünleri yukarı taşıyarak dönüşümü artırmak.

# Yaklaşım

Kullanılan ana kütüphaneler: Polars, Pandas, Catboost

**1) Preprocessing:**
Bu aşamada verilen datadan maksimum verimi alabilmek için oldukça farklı türde özellik çıkarımları uyguladık. Sonrasında sürekli sabit olan veya leak yaratan özelliklerimizi setten çıkardık. Yarattığımız özellik tipleri ve bazı örnekleri:

- Ürün (content) seviyesi toplulaştırmalar (ortalama/medyan/std/cv)
- Kullanıcı seviyesi toplulaştırmalar
- Kullanıcı × Ürün (user–item) toplulaştırmaları
- Oturum-içi (session) özellikleri: min/max/median, rank, z-score, min–max normalize
- Oturum çeşitliliği/entropi, `session_size`, IQR
- Fiyat/indirim/kalite özellikleri: ham, oran, kategori/leaf normalize (z-score), rank
- Dönüşüm oranları: CTR, click→order, cart→order (+ Bayesian smoothing)
- Terim (query) ve terim × ürün: CTR, lift
- Zaman özellikleri: `hour`, `dayofweek`, `elapsed`
- Kullanıcı tercih proxy’leri ve gap’ler
- Arz/rekabet/kıtlık özellikleri
- Unique/duplicate sayımları (`unique_product_cnt`, `dup_product_cnt`)
- Oturum olay sayıları ve oranları (add/remove/view/buy counts & ratios)
- Oturum uzunluğu, `events_per_min`

**2) Validasyon ve Ağırlık Optimizasyonu**
- Zaman bazlı train/eval set ayrımı: Train seti toplam 3 günden, tahmin etmemiz gereken test seti ise bunlardan sonra gelen 1 günden oluşuyor. Bu yüzden validasyon yaparken train setinin ilk 2 günüyle modelimizi eğitip, 3. gün ile başarısını ölçtük. Böylece hem validasyonda yarışma koşullarını birebir taklit etmiş hem de veri sızıntısını önlemiş oluyoruz.

- İki aileli modelleme: Tıklama ve Sipariş için tek bir model oluşturmak yerine modelleri ayrı eğitmeye karar verdik. Böylece iki modeli de ayrı ayrı optimize ederek genel bir skor artışı sağlayabildik.

- Model tercihi CatBoost): Büyük veride yoğun kategorik alanlardan güçlü sinyal çıkarabildiği, hızlı ve kullanımı kolay olduğu için temel model olarak CatBoostRanker) kullandık. CatBoost’un kategorikleri yerleşik olarak işlemesi sayesinde ek karmaşık ön-işleme ihtiyacı azaldı; varsayılan ayarlarla bile istikrarlı sonuç verdi.
> Not: XGBoost ve **LightGBM kurulumu/ayarı (özellikle kategorik hazırlığı ve GPU yapılandırması)*CatBoost’a göre çok daha uğraştırıcıydı. Zaman kısıtı olmasaydı, ikisini de aynı doğrulama düzenine alıp ağırlık optimizasyonuna dahil edecek şekilde akışımıza eklerdik.

- Eğitim seti seçimi: Yarışma metriğine uygun kayıp fonksiyonunun doğru hesaplanabilmesi için, iki ayrı model farklı oturum verileri kullanılarak eğitildi.
  - ORDER modeli: Yalnızca en az bir sipariş içeren oturumlar  
  - CLICK modeli: Sipariş olan oturumlar + yalnızca tıklamalı oturumlar (gerekirse sampling)  

- Ağırlık optimizasyonu: Veri aşırı dengesiz olduğundan, model ağırlıklarını yarışma metriğiyle aynı şekilde belirlemek skor kaybına yol açabiliyordu. Bu nedenle, validasyon seti üzerinde ağırlıkları optimize ederek, genel olarak daha iyi performans veren 0.4–0.6 (click-orer) ağırlıklarını 0.3–0.7 yerine kullandık.

**3) Final Model Eğitimi**
- Son aşamada skorumuzu maksimize etmek için, 2. aşamada elde ettiğimiz modeli temel alarak time series k-fold veya klasik k-fold yerine modelimizi tüm veriyle eğitiyoruz. Böylece hem veri sızıntısı yapmamış oluyoruz hem de modelimiz verinin en önemli kısmı olan son günü ile de eğitilmiş oluyor.




  
