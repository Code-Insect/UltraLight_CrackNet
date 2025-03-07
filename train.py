import torch
from torch.utils.data import DataLoader
# import timm
from tensorboardX import SummaryWriter
from models.UltraLight_CrackNet.UltraLight_CrackNet import UltraLight_CrackNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

from datasets import make_data_loader
from models.Comparative_models.Models import SegNet  # 对比实验模型 -> SegNet
from models.Comparative_models.Mobilenetv3 import MobileNetV3Seg  # 对比实验模型 -> Mobilenetv3
from models.Comparative_models.Efficientnet import EfficientNetSeg, efficientnets  # 对比实验模型 -> Efficientnet
from models.Comparative_models.UNet import UNet  # 对比实验模型 -> UNet
from models.Comparative_models.deeplabv3_plus.deeplab import DeepLab  # 对比实验模型 -> Deeplabv3+（ResNet50）
from models.Comparative_models.CrackSegFormer.CrackSegFormer import SegFormer  # 对比实验模型 -> CrackSegFormer
from models.Comparative_models.linkcrack import LinkCrack  # 对比实验模型 -> LinkCrack
from models.Comparative_models.HrSegNetB48 import HrSegNetB48  # 对比实验模型 -> HrSegNetB48
from models.Comparative_models.TransUNet.vit_seg_modeling import create_TransUNet  # 对比实验模型 -> TransUNet
from models.Comparative_models.DeepCrack import DeepCrackNet

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_loader, val_loader, test_loader = make_data_loader(config)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'Mobilenetv3':
        model = MobileNetV3Seg(48, 576, model_cfg['num_classes'], 'small')  # 对比实验 -> Mobilenetv3
    elif config.network == 'Efficientnet':
        model = EfficientNetSeg(efficientnets[0], 112, 320, model_cfg['num_classes'])  # 对比实验 -> Efficientnet
    elif config.network == 'U_Net':
        model = UNet(3, 1)  # 对比实验 -> UNet
    elif config.network == 'SegNet':
        model = SegNet(num_classes=1)  # 对比实验 -> SegNet
    elif config.network == 'Deeplabv3_plus':
        model = DeepLab()  # 对比实验模型 -> Deeplabv3+（ResNet50）
    elif config.network == 'CrackSegFormer':
        model = SegFormer(num_classes=1)  # 对比实验模型 -> CrackSegFormer
    elif config.network == 'LinkCrack':
        model = LinkCrack()  # 对比实验模型 -> LinkCrack
    elif config.network == 'HrSegNetB48':
        model = HrSegNetB48()  # 对比实验模型 -> HrSegNetB48
    elif config.network == 'DeepCrack':
        model = DeepCrackNet()
    elif config.network == 'TransUNet':
        model = create_TransUNet(img_size=config.input_size_h)  # 对比实验模型 -> TransUNet
    elif config.network == 'UltraLight_CrackNet':
        model = UltraLight_CrackNet(num_classes=model_cfg['num_classes'],
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    else:
        raise Exception('network in not right!')
        
    model = model.cuda()
    cal_params_flops(model, config.input_size_h, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            writer
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_loss': min_loss,
        #         'min_epoch': min_epoch,
        #         'loss': loss,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)