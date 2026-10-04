# Cach 1: Import toan bo thuw vin cua pathlib, cach nay an toan nhuwng rom ra, de goi Path ta phai dung pathlib.Path
# import pathlib

# # Lay file path hien tai
# current_file_path = pathlib.Path(__file__)
# print("duowng dan file hien tai: ",current_file_path)

# Cach 2 la chi lay 1 class Path tu thu vien pathlib de xai cho nngan gon
from pathlib import Path

# Lay file path hien tai
current_file_path = Path(__file__)
print(current_file_path)

# Lay folder chua file hien tai
folder_hien_tai = Path(__file__).parent
print("Folder chua file hien tai: ", folder_hien_tai)

print("CWD: " , Path.cwd())
print("exists: " , Path.exists(folder_hien_tai))

"""
Ghi nhớ:
tên model hay tên thư viện thường là viết thường
cách viết "import <<ten thu viện>> as <<tên viết tắt>>" vd như import numpy á np là cách viết rút gọn lại tên thư viện để cho ngắn gọn
Cách viết from <<tên thư viện>> import <<Class name>>" vd: from pathlib import Path là cách chỉ import một lớp nhỏ (Path) của thư viện pathlib cho tiện sử dụng. 
"""