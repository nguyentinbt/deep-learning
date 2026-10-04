"""
Cấu hình chung cho toàn bộ project.
Tất cả hyperparameters và đường dẫn được tập trung ở đây.
"""
from pathlib import Path

# ==========================================
# ĐƯỜNG DẪN
# ==========================================
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Tạo thư mục nếu chưa tồn tại
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# DATASET
# ==========================================
KAGGLE_DATASET = "atifaliak/youtube-comments-dataset"
CSV_FILENAME   = "YoutubeCommentsDataSet.csv"

# Tên cột trong file CSV (⚠️ kiểm tra lại sau khi load lần đầu)
TEXT_COL  = "Comment"
LABEL_COL = "Sentiment"

# ==========================================
# TIỀN XỬ LÝ
# ==========================================
MIN_FREQ  = 2       # tần suất tối thiểu của từ để đưa vào vocab
MAX_LEN   = 100     # độ dài tối đa của câu (sau padding)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# ==========================================
# CHIA TẬP
# ==========================================
TEST_SIZE = 0.3     # 30% cho temp (val + test)
VAL_SIZE  = 0.5     # trong temp, 50% là val, 50% là test
RANDOM_SEED = 42

# ==========================================
# MÔ HÌNH
# ==========================================
EMBEDDING_DIM = 128
HIDDEN_SIZE   = 128
DROPOUT_RATE  = 0.3
DENSE_UNITS   = 64

# ==========================================
# HUẤN LUYỆN
# ==========================================
EPOCHS      = 15
BATCH_SIZE  = 64
LEARNING_RATE = 1e-3
PATIENCE    = 3     # EarlyStopping