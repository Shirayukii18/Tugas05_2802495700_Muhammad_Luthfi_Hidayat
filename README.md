# Tugas 05 - Advanced CNNs and Transfer Learning

Klasifikasi 10 ras Oxford-IIIT Pet dengan ResNet-18 (B1-B4) dan MobileNetV3-Small (B5), mengacu pada soal dan materi Sesi 05.

## Menjalankan

1. Gunakan Python dengan dependencies `requirements.txt` dan PyTorch CUDA yang mendukung GPU. Run ini memakai torch 2.7.1+cu128 dan torchvision 0.22.1+cu128.
2. Buka `Tugas05_Semua_Kode.ipynb` di Jupyter/VS Code. Pilih kernel `.venv` yang mendukung CUDA.
3. Jalankan semua sel berurutan. Dataset dan bobot diunduh dari sumber resmi saat pertama kali diperlukan.
4. Simpan notebook setelah eksekusi. Run ulang memperbarui `outputs_gpu/`. Cadangkan hasil lama bila diperlukan.

Eksperimen dijalankan secara local pada RTX 5070 Ti. Cache citra, transform, training dan inference memakai CUDA. CPU/RAM digunakan untuk decoding, kontrol, export metrics dan plotting.

## Protokol dan kaitan materi

- 1.981 citra: 1.386 train, 297 validation, 298 test. Stratified 70/15/15, seed 42.
- B1: ResNet dari nol. B2: hanya fc aktif. B3: layer4 dan fc dengan LR 1e-4/1e-3.
- B3 membuka layer4 sejak awal sesuai tabel/kode soal. Gradual unfreezing beberapa tahap pada slide 16 dibahas secara konseptual, tidak diklaim sebagai implementasi.
- B3 dipakai sebagai B4 dengan augmentation, dibandingkan B4 tanpa augmentation dengan configuration sama.
- Statistik BatchNorm pretrained dibekukan. BatchNorm B1 dilatih (slide 6-8).
- Semua ResNet memakai residual connection (slide 10-12). Tidak ada kontrol plain CNN.
- Augmentation hanya train. Materi augmentation terdapat di slide 20-22. Slide 23 adalah Modern Architectures.
- Batch 32, AdamW, cross entropy, maksimum 25 epoch, patience 4. Checkpoint minimum validation loss. Tidak ada scheduler atau Grad-CAM opsional.
- B4 tanpa augmentation dipilih sebelum test. B5 memakai configuration terpilih pada MobileNetV3-Small, alternatif yang diizinkan soal, terkait efisiensi slide 24 dan sumber pretrained slide 25.
- Durasi meliputi training, evaluation validation/train bersih, logging/checkpoint. Tidak termasuk download, initialization dan test akhir.

## Sumber

- Materi Sesi 05: Advanced CNNs and Transfer Learning, Dr. Hidayaturrahman.
- Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- BatchNorm: https://arxiv.org/abs/1502.03167
- ResNet: https://arxiv.org/abs/1512.03385
- Transferability: https://arxiv.org/abs/1411.1792
- MobileNetV3: https://arxiv.org/abs/1905.02244
