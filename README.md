# Trendyol Hackathon 2025 8. Sıra Çözümü

Önemli not: Veri gizliliği kurallarından ötürü yarışma datası repoya dahil edilmemiştir. 

Problem: Bu yarışmada katılımcılar, arama sonuçlarının tıklama ve satın almaya dönüşüm oranlarını artırmaya odaklanır. Hedef; belirli bir kullanıcı × arama terimi × tarih üçlüsü (bir oturum) için her ürünün olası aksiyonlarını (clicked / ordered) dikkate alıp, bu aksiyonların gerçekleşmesi en muhtemel ürünleri öne çıkaracak bir sıralama üretmektir.

Amaç: Her oturumda, ürünlerin tıklanma ve sipariş alma ihtimallerini gözeten bir puanlama ile listeleme kalitesini yükseltmek.

İş hedefi: Kullanıcıya daha alakalı ürünleri yukarı taşıyarak dönüşümü artırmak.

# Yaklaşım

Kullanılan ana kütüphaneler: Polars, Pandas, Catboost

1) Preprocessing: Bu aşamada verilen datadan maksimum verimi alabilmek için oldukça farklı türde özellik çıkarımları uyguladık. Sonrasında sürekli sabit olan veya leak yaratan özelliklerimizi setten çıkardık. Yarattığımız özellik tipleri ve bazı örnekleri:

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



  
