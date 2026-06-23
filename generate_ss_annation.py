import os
import random

from tqdm import tqdm

def create_dataset(path, file_path):
    files = []
    ids = os.listdir(path)
    for id in tqdm(ids):
        id_path = os.path.join(path, id)
        file = [f'{id}/{f}' for f in os.listdir(id_path)]
        files.extend(file)

    random.shuffle(files)

    with open(file_path, 'w') as f:
        for i in files:
            f.write(i + '\n')
    



if __name__ == "__main__":
    path = 'distill_dataset_75_25/train'
    file_path = 'distill_dataset_75_25/train/ss_train.txt'
    create_dataset(path, file_path)