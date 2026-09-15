# import thư viện hõ trợ dường dẫn 

from pathlib import Path


# Get the current file path 
current_file_path = Path(__file__)
print("Current file path is:", current_file_path)

current_dir = current_file_path.parent
print("Current folder which containing file:", current_dir)




