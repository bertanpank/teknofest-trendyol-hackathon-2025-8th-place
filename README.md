# Trendyol Hackathon 2025 8. Sıra Çözümü [TR]

Önemli not: Veri gizliliği kurallarından ötürü yarışma datası repoya dahil edilmemiştir. Kodlar büyük oranda kullanıma hazır haldedir, minör path/kütüphane ayarları gerekebilir.

Problem: Bu yarışmada katılımcılar, arama sonuçlarının tıklama ve satın almaya dönüşüm oranlarını artırmaya odaklanır. Hedef; belirli bir kullanıcı × arama terimi × tarih üçlüsü (bir oturum) için her ürünün olası aksiyonlarını (clicked / ordered) dikkate alıp, bu aksiyonların gerçekleşmesi en muhtemel ürünleri öne çıkaracak bir sıralama üretmektir.

Amaç: Her oturumda, ürünlerin tıklanma ve sipariş alma ihtimallerini gözeten bir puanlama ile listeleme kalitesini yükseltmek.

İş hedefi: Kullanıcıya daha alakalı ürünleri yukarı taşıyarak dönüşümü artırmak.

# Yaklaşım

Kullanılan ana kütüphaneler: Polars, Pandas, Catboost

### 1) Preprocessing
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

### 2) Validasyon ve Ağırlık Optimizasyonu
- Zaman bazlı train/eval set ayrımı: Train seti toplam 3 günden, tahmin etmemiz gereken test seti ise bunlardan sonra gelen 1 günden oluşuyor. Bu yüzden validasyon yaparken train setinin ilk 2 günüyle modelimizi eğitip, 3. gün ile başarısını ölçtük. Böylece hem validasyonda yarışma koşullarını birebir taklit etmiş hem de veri sızıntısını önlemiş oluyoruz.

- İki aileli modelleme: Tıklama ve Sipariş için tek bir model oluşturmak yerine modelleri ayrı eğitmeye karar verdik. Böylece iki modeli de ayrı ayrı optimize ederek genel bir skor artışı sağlayabildik.

- Model tercihi CatBoost): Büyük veride yoğun kategorik alanlardan güçlü sinyal çıkarabildiği, hızlı ve kullanımı kolay olduğu için temel model olarak CatBoostRanker) kullandık. CatBoost’un kategorikleri yerleşik olarak işlemesi sayesinde ek karmaşık ön-işleme ihtiyacı azaldı; varsayılan ayarlarla bile istikrarlı sonuç verdi.
> Not: XGBoost ve LightGBM kurulumu/ayarı (özellikle kategorik hazırlığı ve GPU yapılandırması) CatBoost’a göre çok daha uğraştırıcıydı. Zaman kısıtı olmasaydı, ikisini de aynı doğrulama düzenine alıp ağırlık optimizasyonuna dahil edecek şekilde akışımıza eklerdik.

- Eğitim seti seçimi: Yarışma metriğine uygun kayıp fonksiyonunun doğru hesaplanabilmesi için, iki ayrı model farklı oturum verileri kullanılarak eğitildi.
  - ORDER modeli: Yalnızca en az bir sipariş içeren oturumlar  
  - CLICK modeli: Sipariş olan oturumlar + yalnızca tıklamalı oturumlar (gerekirse sampling)  

- Ağırlık optimizasyonu: Veri aşırı dengesiz olduğundan, model ağırlıklarını yarışma metriğiyle aynı şekilde belirlemek skor kaybına yol açabiliyordu. Bu nedenle, validasyon seti üzerinde ağırlıkları optimize ederek, genel olarak daha iyi performans veren 0.4–0.6 (click-orer) ağırlıklarını 0.3–0.7 yerine kullandık.

### 3) Final Model Eğitimi
- Son aşamada skorumuzu maksimize etmek için, 2. aşamada elde ettiğimiz modeli temel alarak time series k-fold veya klasik k-fold yerine modelimizi tüm veriyle eğitiyoruz. Böylece hem veri sızıntısı yapmamış oluyoruz hem de modelimiz verinin en önemli kısmı olan son günü ile de eğitilmiş oluyor.



# Trendyol Hackathon 2025 — 8th Place Solution [EN]

Important note: Due to data privacy rules, the competition data is not included in the repo. The code is largely ready to use; minor path/library adjustments may be needed.

## Problem
This challenge focuses on increasing the conversion of search results into clicks and purchases. For a given user × query × date tuple (an interaction session), the aim is to rank products by considering the likelihood of each action (clicked / ordered) and surface the most probable items.

Goal: Improve ranking quality by scoring products in each session with both click and order likelihood in mind.  
Business objective: Move more relevant items higher to increase conversions.

# Approach

Main libraries: Polars, Pandas, CatBoost

### 1) Preprocessing
To extract maximum value, we engineered diverse feature families and then removed constant or leakage-prone ones. Examples:

- Product (content)–level aggregations (mean/median/std/cv)  
- User–level aggregations  
- User × Product (user–item) aggregations  
- In-session features: min/max/median, rank, z-score, min–max normalization  
- Session diversity/entropy, `session_size`, IQR  
- Price/discount/quality: raw, ratios, category/leaf-normalized (z-score), rank  
- Conversion rates: CTR, click→order, cart→order (+ Bayesian smoothing)  
- Query and query × product: CTR, lift  
- Time features: `hour`, `dayofweek`, `elapsed`  
- User preference proxies and gaps  
- Supply/competition/scarcity features  
- Unique/duplicate counts (`unique_product_cnt`, `dup_product_cnt`)  
- Session event counts & ratios (add/remove/view/buy)  
- Session length, `events_per_min`

### 2) Validation and Weight Optimization
- Time-based split: Train = Day 1 + Day 2, Validation = Day 3 (mirrors test-day conditions; prevents leakage).  
- Two-model setup: Train CLICK and ORDER separately; optimize each family on its own.  
- Model choice (CatBoost): We used CatBoostRanker for strong signals from high-cardinality categoricals, speed, and ease of use (built-in categorical handling reduces preprocessing).  
  > Note: XGBoost / LightGBM setup (especially for categoricals/GPU) is more involved. With more time, we would add both to the same validation scheme and include them in weight optimization.
- Training cohorts:  
  - ORDER model: sessions with at least one order  
  - CLICK model: sessions with orders plus click-only sessions (optionally sampled)  
- Weight tuning: Due to class imbalance, using competition weights as-is caused loss. We optimized on validation and used 0.4–0.6 (click–order) instead of 0.3–0.7 for better overall performance.

### 3) Final Model Training
In order to maximize final score, we train on all training data (instead of time-series k-fold or classic k-fold) using the model setup from step 2—avoiding leakage while also ensuring the model learns from the most recent day.

  
