from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical

# ==========================================
# 1. CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# ==========================================
print("Đang tải dữ liệu CIFAR-10...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Chuẩn hóa điểm ảnh: Chia cho 255 để đưa giá trị pixel từ (0-255) về khoảng (0-1)
# Việc này giúp mô hình học nhanh và ổn định hơn rất nhiều
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Chuyển đổi nhãn (One-hot Encoding): Biến số 3 thành mảng [0,0,0,1,0,0,0,0,0,0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ==========================================
# 2. XÂY DỰNG KIẾN TRÚC MẠNG CNN
# ==========================================
model = Sequential()

# Đầu vào: Ảnh 32x32 với 3 kênh màu RGB
model.add(Input(shape=(32, 32, 3)))

# Khối Convolution 1
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Khối Convolution 2
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Khối Flatten và Mạng ANN
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))

# Lớp Output: 10 node cho 10 loại đối tượng CIFAR-10
model.add(Dense(units=10, activation='softmax'))

# ==========================================
# 3. CẤU HÌNH VÀ HUẤN LUYỆN (COMPILE & FIT)
# ==========================================
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Bắt đầu huấn luyện mô hình...")
history = model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=32, 
    epochs=10, 
    validation_data=(X_test, y_test)
)

print("Hoàn tất huấn luyện!")