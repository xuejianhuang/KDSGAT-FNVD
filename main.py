import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)
import time
from utils.utils import parse_arguments,set_torch_seed
from utils.logger import set_logger, log_args, log_config
from dataset import getModelAndData
from config import base_config,KDSGAT_FNVD_config,FakingRecipe_config

#===================train==================
def train(model,train_dataloader,criterion,optimizer,metric,epoch,writer):
    """
       Train the model for one epoch.

       Parameters:
       model (nn.Module): The model to train.
       train_dataloader (DataLoader): DataLoader for the training data.
       criterion (nn.Module): Loss function.
       optimizer (Optimizer): Optimizer for training.
       metric (Metric): Metric for evaluation.
       epoch (int): Current epoch number.
       writer (SummaryWriter): SummaryWriter for logging.

       Returns:
       tuple: Training accuracy and loss.
    """
    model.train()
    loss_list = []
    epoch_batch=len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        labels = batch['label']
        optimizer.zero_grad()
        probs = model(batch)
        preds = torch.argmax(probs, dim=-1)

        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        metric.update(preds, labels)
        loss_list.append(loss.cpu().item())

        if step % 10 == 0:
            acc=metric.forward(preds, labels)
            logger.info("epoch: %d, step: %d, loss: %.4f, acc: %.4f" % (epoch, step, loss.item(), acc))
            writer.add_scalar('Loss/train', loss.item(), epoch * epoch_batch + step)
            writer.add_scalar('Acc/train', acc, epoch * epoch_batch + step)
    writer.flush()
    train_loss = np.mean(loss_list)
    train_acc=metric.compute().cpu().numpy()
    metric.reset()
    return train_acc,train_loss

#===================validation==================
@torch.no_grad()
def val(model, metric, criterion, val_dataloader,epoch):
    model.eval()
    loss_list = []
    for batch in val_dataloader:
        labels = batch['label']
        probs = model(batch)
        preds = torch.argmax(probs, dim=-1)

        loss = criterion(probs, labels)
        loss_list.append(loss.cpu().item())
        metric.update(preds, labels)
    val_loss = np.mean(loss_list)
    val_acc = metric.compute().cpu().numpy()
    writer.add_scalar('Loss/val', val_loss,epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    metric.reset()
    return val_acc,val_loss

#===================test==================
@torch.no_grad()
def test(model, acc_metric,pre_metric,rec_metric,f1_metric,confusion, test_dataloader,output_csv):
    """
    Test the model and save misclassified samples to a CSV file.

    Parameters:
    model (nn.Module): The model to test.
    acc_metric (Metric): Accuracy metric.
    pre_metric (Metric): Precision metric.
    rec_metric (Metric): Recall metric.
    f1_metric (Metric): F1 score metric.
    confusion (Metric): Confusion matrix metric.
    test_dataloader (DataLoader): DataLoader for the test data.
    output_csv (str): Path to save the misclassified samples.

    Returns:
    Figure: Confusion matrix plot.
    """
    model.eval()

    acc_metric.reset()
    pre_metric.reset()
    rec_metric.reset()
    f1_metric.reset()
    confusion.reset()

    misclassified_samples = []  # List to store misclassified samples
    for batch in test_dataloader:
        labels = batch['label']
        keys =batch['vid']
        probs = model(batch)
        preds = torch.argmax(probs, dim=-1)
        acc_metric.update(preds, labels)
        pre_metric.update(preds, labels)
        rec_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        confusion.update(preds, labels)

        # Collect misclassified sample key
        for key, pred, label in zip(keys, preds, labels):
            if pred != label:
                misclassified_samples.append([key, label.item(), pred.item()])
    # Convert misclassified samples list to DataFrame
    df = pd.DataFrame(misclassified_samples, columns=['video_id', 'true value', 'predicted value'])

    df.to_csv(output_csv, index=False)

    test_acc = acc_metric.compute().cpu().numpy()
    test_pre = pre_metric.compute().cpu().numpy()
    test_rec = rec_metric.compute().cpu().numpy()
    test_f1 = f1_metric.compute().cpu().numpy()
    fig_, ax_ = confusion.plot()
    confusion_matrix =confusion.compute().cpu().numpy()
    logger.info('--------------------- test results-------------------------------')
    logger.info('acc:' + str(test_acc) + '  prec:' + str(test_pre) +
                '  rec:' + str(test_rec) + '  f1:' + str(test_f1))
    logger.info(f'confusion: \n{confusion_matrix}')

    return fig_


if __name__ == '__main__':
    args =parse_arguments()
    logger = set_logger(args)
    log_args(logger,args)
    if args.model=='KDSGAT-FNVD':
        set_torch_seed(KDSGAT_FNVD_config.seed)
        log_config(logger, KDSGAT_FNVD_config)
    else:
        set_torch_seed(FakingRecipe_config.seed)
        log_config(logger, FakingRecipe_config)



    writer=SummaryWriter()

    model,train_dataloader,val_dataloader,test_dataloader = getModelAndData(model_name=args.model,dataset=args.dataset)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('total number of parameters:{}'.format(pytorch_total_trainable_params))

    start_time = time.time()
    criterion = CrossEntropyLoss()
    # Metric for evaluation during training
    acc_metric = MulticlassAccuracy(num_classes=base_config.num_classes).to(base_config.device)
    optimizer = AdamW(model.parameters(), lr=base_config.lr)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=base_config.decayRate)
    best_acc = 0
    early_stop_cnt = 0
    model_saved_path = base_config.model_saved_path + f"{args.model}_{args.dataset}" + ".pkl"
    if not os.path.exists(base_config.model_saved_path):
        os.makedirs(base_config.model_saved_path)
    # Save misclassified samples during testing
    misclassified_save_path= args.model + "_" + args.dataset+"_misclassified.csv"

    if args.mode in ['train', 'both']:
        for epoch in range(base_config.epoch):
            logger.info('-----------Epoch:' + str(epoch) + "-----------")
            train_acc, train_loss = train(model,train_dataloader,criterion,optimizer,acc_metric,epoch,writer)
            logger.info('train_loss:{:.5f} train_acc:{:.3f}'.format(train_loss, train_acc))
            val_acc, val_loss = val(model, acc_metric, criterion, val_dataloader,epoch)
            logger.info("val_loss:{:.5f} val_acc:{:.3f} \n".format(val_loss, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_saved_path)
                logger.info("save model,acc:{:.3f}".format(best_acc))
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt >= base_config.patience:
                break
            lr_scheduler.step()
    if args.mode in ['test', 'both']:
        pre_metric=MulticlassPrecision(num_classes=base_config.num_classes,average=None).to(base_config.device)
        rec_metric=MulticlassRecall(num_classes=base_config.num_classes,average=None).to(base_config.device)
        f1_metric=MulticlassF1Score(num_classes=base_config.num_classes,average=None).to(base_config.device)
        confusion=MulticlassConfusionMatrix(num_classes=base_config.num_classes).to(base_config.device)
        model.load_state_dict(torch.load(model_saved_path))
        confusion_matrix=test(model, acc_metric,pre_metric,rec_metric,f1_metric,confusion, test_dataloader,misclassified_save_path)
        confusion_matrix.savefig(args.model + "_" + args.dataset+'_confusion_matrix.png')
    writer.close()
    end_time = time.time()
    logger.info("the running time is: {:.1f} s".format(end_time - start_time))