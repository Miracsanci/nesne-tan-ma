# nesne-tanima
Yapay Zeka ile Nesne tanımlama Projesi

Bu proje, ResNet50 mimarisi ve ImageNet sınıflandırma modeli kullanarak yüklenen görsellerdeki alet ve elektronik eşya türlerini tanımlar. Kullanıcılar, Gradio arayüzü sayesinde bir görsel yükleyerek modelin tahminlerini hızlıca görebilir.

📌 Özellikler
Görseldeki alet ve eşya nesnelerini tanır.
PyTorch'un önceden eğitilmiş ResNet50 modeliyle çalışır.
Gradio arayüzü ile kullanıcı dostu bir deneyim sunar.
En yüksek olasılıklı ilk 5 sınıf gösterilir.
Sadece belirli alet ve cihaz sınıfları filtrelenerek sunulur.

🧰 Kullanılan Teknolojiler
Python 3.x
PyTorch
Torchvision
Gradio
Pillow (PIL)
NumPy

⚙️ Kurulum
pip install torch torchvision gradio pillow numpy requests ile  kütüphaneleri yükleyin
python app.py ile uygulamayı başlatın

🖼 Kullanım
Arayüzde "Bir görsel yükle" bölümünden bir resim seçin.
"🔍 Sınıflandır" butonuna tıklayın.
Tahmin edilen en olası sınıf ve ilk 5 sınıf listelenir.

🎯 Hedeflenen Sınıflar
screwdriver, power drill, hand blower, vacuum, microwave, refrigerator, 
laptop, notebook, desktop computer, monitor, keyboard, mouse, 
cellular telephone, iPod, remote control, camera, digital clock, 
printer, projector, torch, electric fan, table lamp
