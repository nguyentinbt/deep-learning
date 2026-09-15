# Dau vao la danh sach cac tham so loss cua mo hinh
loss_values = [0.8, 0.5, 0.3, 0.1]

# Duyet qua cac phan tu trong danh sach

for loss in loss_values:
    #  Dau ra in ra tung gia tri
    print("Sai so hien tai la",loss)

# dau vao la mot danh sach cac xac suat
prediction = [0.1, 0.9, 0.3, 0.8]

# Xu ly: Duyet qua tung bien p trong predict, gan gia tri 1 neu lonw hon 0.5, nguowjc laij gan 0

labels = [1 if p > 0.5 else 0 for p in prediction]

print(labels)

# ta co danh sach da du doan
prediction_list = [0.1, 0.4, 0.5, 0.6, 0.7]


# Xu ly danh sach bawng cang new p > 0.5 thi cap nhat 
new_labels = [1 if p > 0.6 else 0 for p in prediction_list]
print(new_labels)



pixels = [255, 127, 0, 50]

normalized_pixels = [p/255 for p in pixels]
print(normalized_pixels)
