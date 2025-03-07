from datasets import Steelcrack_dataset
from torch.utils.data import DataLoader


def make_data_loader(config):
    if config.datasets == 'SteelCrack':
        train_set = Steelcrack_dataset.CrackSegmentation(split='train')
        val_set = Steelcrack_dataset.CrackSegmentation(split='val')
        test_set = Steelcrack_dataset.CrackSegmentation(split='test')
    elif config.datasets == 'DeepCrack':
        train_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/DeepCrack_voc", split='train')
        val_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/DeepCrack_voc", split='val')
        test_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/DeepCrack_voc", split='test')
    elif config.datasets == 'Crack500':
        train_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/CRACK500_voc", split='train')
        val_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/CRACK500_voc", split='val')
        test_set = Steelcrack_dataset.CrackSegmentation(base_dir=r"../autodl-tmp/CRACK500_voc", split='test')
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader


"""
测试dataloader功能
"""
class config:
    batch_size = 8
    num_workers = 10


if __name__ == '__main__':
    train_loader, val_loader, test_loader = make_data_loader(config)
    print()