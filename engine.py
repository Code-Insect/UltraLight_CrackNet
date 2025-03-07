import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(tqdm(train_loader)):
        step += iter
        optimizer.zero_grad()
        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    if scheduler is not None:
        scheduler.step()  # 更新学习率
    return step


def val_one_epoch(val_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    writer):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data['image'], data['label']
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # y_pre = np.where(preds>=config.threshold, 1, 0)
        # y_true = np.where(gts>=0.5, 1, 0)
        y_pre = (preds >= config.threshold).astype(int)
        y_true = (gts >= 0.5).astype(int)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0  # 召回率
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0  # 精确率
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0  # 特异度：预测正确的所有负样本占实际所有负样本的比例
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                specificity: {specificity:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        writer.add_scalar('val/loss', np.mean(loss_list), epoch)
        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/F1_score', f1_or_dsc, epoch)
        writer.add_scalar('val/Recall', recall, epoch)
        writer.add_scalar('val/Precision', precision, epoch)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data['image'], data['label']
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # y_pre = np.where(preds>=config.threshold, 1, 0)
        # y_true = np.where(gts>=0.5, 1, 0)
        y_pre = (preds >= config.threshold).astype(int)
        y_true = (gts >= 0.5).astype(int)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0  # 精确率
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                specificity: {specificity:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

