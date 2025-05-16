import xml.etree.ElementTree as ET
import os
import shutil
from natsort import natsorted
from tqdm import tqdm

for data_split in ["train", "valid", "test"]:
    print(f"Processing {data_split} split...")
    with open(f"./book_split/books_{data_split}.txt", "r") as book_list:
        manga_list = book_list.readlines()

    os.makedirs(f"./{data_split}/", exist_ok=True)

    for manga_name in tqdm(natsorted(manga_list), desc=f"{data_split} books"):
        manga_name = manga_name.strip()

        tree = ET.parse(f"./Manga109_released_2023_12_07/annotations/{manga_name}.xml")
        root = tree.getroot()

        for page in root.iter("page"):
            page_index = page.attrib["index"]
            page_index_zfill3 = page_index.zfill(3)
            page_index_zfill6 = page_index.zfill(6)
            src = f"./Manga109_released_2023_12_07/images/{manga_name}/{page_index_zfill3}.jpg"
            dst = f"./{data_split}/{manga_name}_{page_index_zfill6}.jpg"

            # Skip if already exists
            if os.path.exists(dst):
                continue

            shutil.copyfile(src, dst)
