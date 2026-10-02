# YOLO Gercek Zamanli Nesne Tespiti ve Takibi
Webcam goruntusu uzerinde YOLO ile nesne tespiti ve takibi yapan basit bir proje.

Nesnelerin etrafina kutu cizilir ve nesne adi, confidence degeri ve takip ID'si gosterilir. Ayrica ekranda FPS degeri bulunur.

Projede CPU ve NVIDIA GPU icin iki farkli surum bulunuyor.

CPUrun:
* yolo11n modeli kullanir.
* CPU ile calisir.
* Yaklasik 13 FPS test edildi.

CUDAGPUrun:
* yolo11m modeli kullanir.
* NVIDIA GPU ve CUDA ile calisir.
* Yaklasik 26 FPS test edildi.
* Nesne ID'lerine gore renk ve nesne sayaci bulunur.

## Kurulum

CPUrun klasorunde

cd CPUrun
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

CUDAGPUrun icin

cd CUDAGPUrun
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

## Calistirma
Ilgili klasörde

venv\Scripts\activate
python main.py

Program acildiktan sonra Q tusuna basarak cikabilirsiniz

Model dosyalari ilk calistirmada otomatik olarak indirilir

## Ayarlar
Kamera numarasi, cozunurluk, confidence degeri, model ve takip algoritmasi config.json dosyasindan degistirilebilir.

CUDAGPUrun icin NVIDIA GPU ve CUDA destekli PyTorch gereklidir.

## Iletisim
Atakan Koçoğlu -atakankocoglu.iletisim@gmail.com
