import pandas as pd
import numpy as np
data = {
    "ma_kh": ["KH01","KH02","KH03","KH04"],
    "tuoi": [35, -5, 42,28],
    "r_churn": [0,1,1,0]
}

df = pd.DataFrame(data)
print(df)

# Xử lý: Lọc lấy những khách hàng có tuổi hợp lệ (lớn hơn 0)
# df["tuoi"] > 0 sẽ tạo ra một danh sách [True, False, True, True] 
# Pandas sẽ chỉ giữ lại những hàng mang giá trị True

ds_tuoi = df["tuoi"]
print("Tuoi khach hang: ", ds_tuoi)

# Xu ly voi numpy
check_tuoi = [True if tuoi > 0 else False for tuoi in ds_tuoi]
print(check_tuoi)
df_valid_tuoi = df[check_tuoi]
print("Valid tuoi: ", df_valid_tuoi)

# Xu ly voiw pandas
df_clean = df[df["tuoi"]>0]
print(df_clean)

# Loc lay nguoi da bo dich vu r_churn ==1

df_churn = df_clean[df_clean["r_churn"]==1]
print("Khach hang da roi di : ", df_churn)