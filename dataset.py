import torch.utils.data as data
from PIL import Image, ImageFile
import os
import csv

ImageFile.LOAD_TRUNCATED_IAMGES = True


def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

def csv_reader(fileList):
    dataPath = "/media/data4/zhangzhenduo/dataset/traindataset/CASIA_aligned"
    imgList = []
    with open(fileList, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            dirname = os.path.join(dataPath, row[1])
            imgPath = os.path.join(dirname, row[0]+'.jpg')
            label = row[3]
            imgList.append((imgPath, int(label)))
    
    return imgList


class ImageList(data.Dataset):
    '''
     Args:
        root (string): Root directory path.
        fileList (string): Image list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    '''

    def __init__(self, root, fileList, transform=None, list_reader=csv_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)