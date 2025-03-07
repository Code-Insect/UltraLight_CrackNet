from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
# from mypath import Path
from torchvision import transforms
from datasets import custom_transforms as tr


class CrackSegmentation(Dataset):
    """
    Crack dataset
    """

    def __init__(self,
                 base_dir=r"../autodl-tmp/Steelcrack_voc",  # Steelcrack数据集路径
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val/test
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split


        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                if 'Steelcrack' in self._base_dir:
                    _image = os.path.join(self._image_dir, line + '.png')  # SteelCrack为png
                else:
                    _image = os.path.join(self._image_dir, line + '.jpg')
                _cat = os.path.join(self._cat_dir, line + '.png')

                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
            elif split == 'test':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):  # 对训练集的预处理
        if 'Steelcrack' in self._base_dir:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),  # 随机水平翻转
                tr.RandomVerticalFlip(),  # 随机垂直翻转
                tr.RandomRotate(degree=np.random.randint(15, 30)),  # 随机旋转
                tr.Normalize(),
                tr.ToTensor(),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=384),  # 中心裁剪为384×384，针对DeepCrack和Crack500
                tr.RandomHorizontalFlip(),  # 随机水平翻转
                tr.RandomVerticalFlip(),  # 随机垂直翻转
                tr.RandomRotate(degree=np.random.randint(15, 30)),  # 随机旋转
                tr.Normalize(),
                tr.ToTensor(),
            ])

        return composed_transforms(sample)

    def transform_val(self, sample):  # 对验证集/测试集的预处理
        if 'Steelcrack' in self._base_dir:
            composed_transforms = transforms.Compose([
                tr.Normalize(),
                tr.ToTensor(),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=384),  # 中心裁剪为384×384，针对DeepCrack和Crack500
                tr.Normalize(),
                tr.ToTensor(),
            ])

        return composed_transforms(sample)


    def __str__(self):
        return '(split=' + str(self.split) + ')'



