# Tugas 05 - Advanced CNNs and Transfer Learning

Muhammad Luthfi Hidayat | NIM 2802495700

Repository public: https://github.com/Shirayukii18/Tugas05_2802495700_Muhammad_Luthfi_Hidayat

Klasifikasi 10 ras Oxford-IIIT Pet dengan ResNet-18 (B1-B4) dan MobileNetV3-Small (B5), mengacu pada soal dan materi Sesi 05.

## Berkas utama

- `Tugas05_Semua_Kode.ipynb`: seluruh kode; 36 sel kode dieksekusi tanpa error.
- `output/pdf/Laporan_Tugas05_2802495700_Muhammad_Luthfi_Hidayat.pdf`: laporan lima halaman, Times New Roman 14 pt.
- `outputs_gpu/metrics.csv` dan `.json`: metrik akhir.
- `outputs_gpu/B*/history.csv`: log per epoch; `test_predictions.csv`: prediksi test.
- `outputs_gpu/split_manifest.csv`: pembagian data dan identitas citra.
- `outputs_gpu/selection.json`: pemilihan model sebelum test.
- `outputs_gpu/environment.json`: konfigurasi lingkungan.
- `outputs_gpu/failure_analysis.csv`: enam kesalahan; hipotesis visual ada di laporan.
- `requirements.txt` dan `requirements-local-lock.txt`: dependensi dan versi lokal.

Folder `outputs/` serta notebook lama bukan sumber angka laporan terbaru.

## Menjalankan

1. Gunakan Python dengan dependensi `requirements.txt` dan PyTorch CUDA yang mendukung GPU. Run ini memakai torch 2.7.1+cu128 dan torchvision 0.22.1+cu128.
2. Buka `Tugas05_Semua_Kode.ipynb` di Jupyter/VS Code; pilih kernel `.venv` yang mendukung CUDA.
3. Jalankan semua sel berurutan; dataset dan bobot diunduh dari sumber resmi saat pertama kali diperlukan.
4. Simpan notebook setelah eksekusi. Run ulang memperbarui `outputs_gpu/`; cadangkan hasil lama bila diperlukan.

Run memakai RTX 5070 Ti lokal sesuai permintaan pengguna, bukan Colab T4 yang disebut soal. Cache citra, transformasi, training dan inferensi memakai CUDA. CPU/RAM tetap diperlukan untuk decoding, kontrol, ekspor metrik dan plotting. Waktu lokal tidak setara waktu Colab.

## Protokol dan kaitan materi

- 1.981 citra: 1.386 train, 297 validation, 298 test; stratified 70/15/15, seed 42.
- B1: ResNet dari nol; B2: hanya fc aktif; B3: layer4 dan fc dengan LR 1e-4/1e-3.
- B3 membuka layer4 sejak awal sesuai tabel/kode soal. Gradual unfreezing beberapa tahap pada slide 16 dibahas secara konseptual, tidak diklaim sebagai implementasi.
- B3 dipakai sebagai B4 dengan augmentasi, dibandingkan B4 tanpa augmentasi dengan konfigurasi sama.
- Statistik BatchNorm pretrained dibekukan; BatchNorm B1 dilatih (slide 6-8).
- Semua ResNet memakai residual connection (slide 10-12); tidak ada kontrol plain CNN.
- Augmentasi hanya train. Materi augmentasi terdapat di slide 20-22; slide 23 adalah Modern Architectures.
- Batch 32, AdamW, cross entropy, maksimum 25 epoch, patience 4. Checkpoint minimum validation loss; tidak ada scheduler atau Grad-CAM opsional.
- B4 tanpa augmentasi dipilih sebelum test. B5 memakai konfigurasi terpilih pada MobileNetV3-Small, alternatif yang diizinkan soal, terkait efisiensi slide 24 dan sumber pretrained slide 25.
- Durasi meliputi training, evaluasi validation/train bersih, logging/checkpoint; tidak termasuk download, inisialisasi dan test akhir.

## Status pengumpulan dan AI

Repository GitHub bersifat public; belum ada pengiriman ke LMS. ZIP tidak menyertakan dataset, lingkungan virtual atau checkpoint besar. Ketentuan Colab berbeda dari run lokal yang diminta pengguna.

AI membantu saya mengerjakan dan memahami code. Alat: OpenAI Codex. AI juga membantu menyusun laporan. Verifikasi mandiri yang dikonfirmasi mahasiswa: menjalankan notebook, mempelajari code, mencocokkan metrik. Soal meminta analisis dan kesimpulan ditulis mandiri; tinjau dan rumuskan kembali sesuai pemahaman sebelum pengumpulan.

## Sumber

- Materi Sesi 05: Advanced CNNs and Transfer Learning, Dr. Hidayaturrahman.
- Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- BatchNorm: https://arxiv.org/abs/1502.03167
- ResNet: https://arxiv.org/abs/1512.03385
- Transferability: https://arxiv.org/abs/1411.1792
- MobileNetV3: https://arxiv.org/abs/1905.02244
