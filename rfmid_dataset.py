import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RFMiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 第一欄通常是 ID
        self.id_col = self.labels.columns[0]

        # 第二欄通常是 Disease_Risk，不拿來當分類 label
        # 真正的疾病 one-hot 類別從第三欄開始
        self.label_cols = list(self.labels.columns[2:])

        if len(self.label_cols) == 0:
            raise ValueError("No label columns found. Please check the CSV format.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # 圖片 ID -> 檔名
        img_id = str(row[self.id_col])
        if not img_id.lower().endswith((".png", ".jpg", ".jpeg")):
            img_name = img_id + ".png"
        else:
            img_name = img_id

        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # one-hot -> class index
        label_values = row[self.label_cols].astype(float).values
        label = int(label_values.argmax())

        if self.transform:
            image = self.transform(image)

        return image, label