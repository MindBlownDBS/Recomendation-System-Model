# Gunakan base image Python yang resmi
# Pilih versi Python yang sesuai dengan pengembangan Anda (mis. 3.9, 3.10, 3.11)
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan Docker cache
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari direktori lokal Anda ke direktori kerja di dalam container
# Termasuk app.py, model .keras, file .csv, dan file .json
COPY . .

# Set environment variable untuk Flask (opsional, tapi baik untuk pengembangan)
# Untuk produksi, Anda mungkin ingin mengaturnya secara berbeda atau menggunakan Gunicorn
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_DEBUG=1 # Nyalakan hanya untuk debugging, matikan untuk produksi

# Expose port yang akan digunakan oleh aplikasi Flask Anda
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask saat container dimulai
# Menggunakan server development Flask bawaan.
# Untuk produksi, SANGAT disarankan menggunakan production-ready WSGI server seperti Gunicorn.
CMD ["flask", "run"]

# --- CONTOH JIKA MENGGUNAKAN GUNICORN UNTUK PRODUKSI ---
# Pastikan gunicorn ada di requirements.txt Anda
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
# -------------------------------------------------------