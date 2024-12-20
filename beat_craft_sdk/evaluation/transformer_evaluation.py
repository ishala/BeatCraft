from pyemd import emd
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

####### EMD Evaluation #########
# Fungsi untuk merekonstruksi pitch pattern dari data validasi
def reconstruct_pitch_patterns(valDf):
    # Gabungkan pitch_pattern berdasarkan judul lagu dengan mempertimbangkan urutan bar
    return (
        valDf
        .sort_values(by=['judul_lagu', 'bar_number'])  # Pastikan bar diurutkan
        .groupby('judul_lagu')['bar']  # Gunakan kolom 'bar' untuk pitch pattern
        .apply(lambda x: ' '.join(x))  # Gabungkan pitch pattern dari setiap bar
        .reset_index()
    )

def normalize_distribution(distribution):
    total = sum(distribution)
    return [x / total if total > 0 else 0 for x in distribution]

# Fungsi untuk mengonversi urutan pitch menjadi distribusi pitch
def convert_to_pitch_distribution(sequence):
    # Jika sequence adalah list, gabungkan menjadi string
    if isinstance(sequence, list):
        # Konversi setiap elemen menjadi string jika elemen berupa tuple
        sequence = ' '.join(map(str, sequence))
    
    # Ekstrak nada (A-G) menggunakan regex
    pitch_set = 'ABCDEFG'
    pitches = re.findall(f"[{pitch_set}]", sequence)
    
    # Hitung distribusi pitch
    return [pitches.count(p) for p in pitch_set]

# Fungsi untuk menghitung jarak EMD
def calculate_emd(validasi_distribution, generated_distribution):
    # Konversi semua distribusi menjadi np.float64
    validasi_distribution = np.array(validasi_distribution, dtype=np.float64)
    generated_distribution = np.array(generated_distribution, dtype=np.float64)

    # Pastikan kedua distribusi memiliki panjang yang sama
    if len(validasi_distribution) != len(generated_distribution):
        raise ValueError(f"Distribusi tidak sama panjang: validasi={len(validasi_distribution)}, generated={len(generated_distribution)}")

    # Matriks jarak (harus berupa float64)
    distance_matrix = np.abs(np.subtract.outer(range(len(validasi_distribution)), range(len(generated_distribution)))).astype(np.float64)
    
    # Hitung EMD
    return emd(validasi_distribution, generated_distribution, distance_matrix)

# Fungsi evaluasi utama dengan EMD
def evaluate_with_EMD(valDf, generatedSeq):
    reconstructed_data = reconstruct_pitch_patterns(valDf=valDf)
    
    print('generatedSeq', generatedSeq)
    # Konversi distribusi pitch hasil generate
    generated_distribution = convert_to_pitch_distribution(generatedSeq)
    if sum(generated_distribution) == 0:
        raise ValueError("Distribusi pitch hasil generate kosong.")

    results = []
    invalid_data = []  # Simpan baris dengan distribusi kosong
    for idx, row in reconstructed_data.iterrows():
        judul = row['judul_lagu']
        validasi_sequence = row['bar']  # Ambil kolom 'bar' untuk evaluasi pitch

        # Konversi distribusi pitch data validasi
        validasi_distribution = normalize_distribution(convert_to_pitch_distribution(validasi_sequence))
        generated_distribution = normalize_distribution(generated_distribution)

        if sum(validasi_distribution) == 0:
            invalid_data.append({'judul_lagu': judul, 'validasi_sequence': validasi_sequence})
            continue  # Lewati baris ini tanpa error

        # Hitung EMD
        emd_distance = calculate_emd(validasi_distribution, generated_distribution)

        results.append({
            'judul_lagu': judul,
            'emd_distance': emd_distance,
            'validasi_sequence': validasi_sequence
        })
    
    results = pd.DataFrame(results)
    # Log data tidak valid untuk analisis lebih lanjut
    if invalid_data:
        invalid_df = pd.DataFrame(invalid_data)
        invalid_df.to_csv("assets/invalid_pitch_data.csv", index=False)
        print(f"Data dengan distribusi pitch kosong disimpan di 'invalid_pitch_data.csv'.")

    return results

# Function to calculate descriptive statistics
def calculate_statistics(data):
    mean_emd = data['emd_distance'].mean()
    median_emd = data['emd_distance'].median()
    max_emd = data['emd_distance'].max()
    min_emd = data['emd_distance'].min()

    stats = {
        "mean": mean_emd,
        "median": median_emd,
        "max": max_emd,
        "min": min_emd
    }

    print(f"Mean EMD Distance: {mean_emd:.2f}")
    print(f"Median EMD Distance: {median_emd:.2f}")
    print(f"Max EMD Distance: {max_emd:.2f}")
    print(f"Min EMD Distance: {min_emd:.2f}")
    return stats

# Function to identify outliers
def identify_outliers(data):
    outlier_song = data.loc[data['emd_distance'].idxmax()]
    print(f"Lagu dengan EMD tertinggi:\n{outlier_song}")
    return outlier_song

# Function to plot histogram
def plot_histogram(data, column='emd_distance', bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribusi EMD Distance')
    plt.xlabel('EMD Distance')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot boxplot
def plot_boxplot(data, column='emd_distance'):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[column], vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    plt.title('Boxplot EMD Distance')
    plt.xlabel('EMD Distance')
    plt.show()

# Function to plot scatter plot
def plot_scatter(data, column='emd_distance', mean_value=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data[column], color='blue', alpha=0.7)
    if mean_value is not None:
        plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.title('Scatter Plot EMD Distance per Song')
    plt.xlabel('Song Index')
    plt.ylabel('EMD Distance')
    if mean_value is not None:
        plt.legend()
    plt.show()