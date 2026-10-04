import pandas as pd
import numpy as np

# Dau vao: Dua lieu co chua gia tri trong

data = {
    "ma_kh": ["KH01","KH02","KH03"],
    "tuoi": [35, np.nan, 28]
}

print(data)

# convert qua DataFrame bawng thu vien pandas

df = pd.DataFrame(data)
print(df)

# Xu ly cac gia tri nan 
df_filled = df.fillna(0)
print(df_filled)


# # chuyen cot tuoi sng array
# tuoi_array = np.array(df_filled["tuoi"])
# trung_binh_tuoi = np.mean(tuoi_array)
# print(trung_binh_tuoi)

# tuoi_array_updated = [trung_binh_tuoi if tuoi == 0 else tuoi for tuoi in tuoi_array]
# print(tuoi_array_updated)

# df_filled = df_filled[df_filled["tuoi"]> 0]

# print("Sau khi update tuoi", df_filled)

# Update gia tri tuoi 0 thanh mean

mean_age = df["tuoi"].mean()
df_clean_age = df["tuoi"].fillna(mean_age)
print(df_clean_age)

# Thay the tuoi = 0 thanh tuoi = mean
print("Before: ", df_filled)
df_filled["tuoi"] = df_filled["tuoi"].replace(0, mean_age)
print("After : ", df_filled)