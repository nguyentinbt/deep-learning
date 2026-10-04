import pandas as pd
from pathlib import Path

# Dau vao la file du lieu
file_path = Path(__file__).parent.parent/"data/customer_churn.csv"
print(file_path)

df = pd.read_csv(file_path)
print(df)
"""
Ghi nhớ: Với pandas, đr chuyển 1 dữ liệu thì dâtframe ta có các cách sau
- Với dữ lệlaf Dictionart ta dùng keywork DataFrame để convert. VD: pd.DataFrame(dictionary_name) 
- Với file CSV, ta dùng keyword read_CSV. pd.Read_CSV(file_path) 
"""

# Lay mot cot dư lei
gioi_tinh = df["gender"]
print("Cot gioi tinh:\n", gioi_tinh)
