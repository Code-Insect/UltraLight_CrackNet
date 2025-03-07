from utils import *
from engine import *
import torch
from models.UltraLight_CrackNet.UltraLight_CrackNet import UltraLight_CrackNet
from datasets import make_data_loader
from configs.config_setting import setting_config


def test_one_epoch(test_loader,
                    model,
                    criterion,):
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

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = (preds >= 0.5).astype(int)
        y_true = (gts >= 0.5).astype(int)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0  # 精确率
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                specificity: {specificity:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, confusion_matrix: {confusion}'
        print(log_info)

    return np.mean(loss_list)

if __name__ == '__main__':
    config = setting_config
    model = UltraLight_CrackNet(num_classes=1,
                                       input_channels=3,
                                       c_list=[8, 16, 24, 32, 48, 64],
                                       split_att='fc',
                                       bridge=True, )
    model = model.cuda()
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=0)


    train_loader, val_loader, test_loader = make_data_loader(config)
    criterion = config.criterion

    best_weight = torch.load(r'../autodl-tmp/SteelCrack_Proposed.pth', map_location=torch.device('cpu'))
    new_weight_dict = {}
    for key, value in best_weight.items():
        if "total_ops" not in key and "total_params" not in key:
            new_weight_dict[key] = value
    model.load_state_dict(new_weight_dict)
    loss = test_one_epoch(
                test_loader,
                model,
                criterion,)
    print(loss)



