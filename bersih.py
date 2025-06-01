import pandas as pd

# Load dataset dari CSV
df = pd.read_csv(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\DATASET\dataset_final_bersih.csv")

# Simpan dataset ke file pickle (.pkl)
df.to_pickle(r"D:\SEMESTER 6\PENGGALIAN DATA\DASHBOARD\FILE_PKL\dataset_final_bersih.pkl")

print("Konversi CSV ke PKL selesai!")
