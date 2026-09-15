import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(dataroot="data/train", batch_size=128, image_size=64, workers=2):
    """
    Khởi tạo DataLoader để đọc ảnh từ thư mục.
    
    Args:
        dataroot (str): Đường dẫn đến thư mục chứa dữ liệu (cần chứa thư mục con, vd: data/train/images)
        batch_size (int): Số lượng ảnh trong một batch
        image_size (int): Kích thước ảnh (mặc định 64x64 cho DCGAN)
        workers (int): Số luồng CPU dùng để tải dữ liệu
    """
    # Định nghĩa các bước tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # Chuẩn hóa giá trị pixel từ [0, 1] về khoảng [-1, 1] 
        # Điều này rất quan trọng vì lớp cuối cùng của Generator dùng hàm Tanh()
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load dataset từ folder bằng ImageFolder
    # Lưu ý: PyTorch ImageFolder yêu cầu cấu trúc thư mục: dataroot/class_name/xxx.jpg
    # Nghĩa là biến dataroot phải trỏ đến 'data/train', và bên trong có thư mục 'images'
    dataset = dset.ImageFolder(root=dataroot, transform=transform)
    
    # Khởi tạo DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=workers)
    
    return dataloader