# 🤖 PyTorch & Tkinter Masaüstü Yapay Zeka Chatbot

Bu proje, **PyTorch** derin öğrenme kütüphanesi kullanılarak eğitilmiş ve **Tkinter** ile modern bir masaüstü arayüzüne kavuşturulmuş bir yapay zeka chatbot uygulamasıdır. 

Geleneksel CUDA (Nvidia) zorunluluğunun aksine, **Torch-DirectML** entegrasyonu sayesinde AMD ve Intel grafik kartlarında da donanım ivmeli (GPU) olarak çalışabilmektedir.

## ✨ Özellikler
- **Doğal Dil İşleme (NLP):** NLTK kütüphanesi ile metin temizleme, tokenization ve Bag of Words işlemleri.
- **Yapay Sinir Ağları:** Tam bağlantılı (Fully Connected) doğrusal sinir ağı mimarisi.
- **DirectML Desteği:** AMD, Intel ve Nvidia ekran kartlarında yüksek performanslı çalışma.
- **Kullanıcı Dostu Arayüz:** Tkinter ile geliştirilmiş, otomatik kaydırma özellikli mesajlaşma ekranı.

## 📁 Proje Yapısı
- `app.py`: Uygulamanın Tkinter tabanlı görsel arayüzü.
- `chat.py`: Gelen mesajları işleyen ve DirectML üzerinde modeli çalıştıran ana mekanizma.
- `model.py`: NeuralNet sınıfının ve sinir ağı mimarisinin bulunduğu dosya.
- `nltk_utils.py`: NLP fonksiyonlarını (tokenize, stem, bag_of_words) içeren yardımcı dosya.
- `train.py`: `intents.json` verileriyle modeli eğiten ve `data.pth` çıktısını veren eğitim kodu.
- `intents.json`: Botun anlayabileceği şablonları ve vereceği cevapları içeren veri seti.

## 🚀 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin.
2. Terminalde sanal ortamınızı (venv) aktif edin.
3. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
