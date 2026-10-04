import numpy as np
import pandas as pd

dataset_info = {
    "Loai_Vat_The": ["Chó", "Mèo", "Chim"],
    "So_Luong_Anh": [1500, 1200, 800]
}

# Chuyen Dictionary thanh DataFrame de su dung trong pandas
df_dataset = pd.DataFrame(dataset_info)
print("Data frame vua tao: ", df_dataset)

# Lay mot cot tu DataFrame
cot_Loai_Vat_The = df_dataset["Loai_Vat_The"]
print("Cot Loai_Vat_The chua cac gia tri: ", cot_Loai_Vat_The)