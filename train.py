# import argparse
import glob
import os
import pickle
import time

import nibabel as nib
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, write_nifti
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceFocalLoss
from monai.metrics import compute_hausdorff_distance, compute_meandice, DiceMetric, get_confusion_matrix, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import monai.transforms as tf
from monai.utils import get_torch_version_tuple, set_determinism
import pandas as pd
import torch
import numpy as np

print_config()

if get_torch_version_tuple() < (1, 6):
    raise RuntimeError('AMP feature only exists in PyTorch version greater than v1.6.')

# set AMP
amp = True

################################################################################
### ARGPARSE SETUP
################################################################################

# parser = argparse.ArgumentParser(description='Train 3D Res U-Net with contrast (compression) augmentation.')
# parser.add_argument('--prob', nargs=1, type=float,
#                     help='Global probability that the transform gets applied during augmentation.')
# args = parser.parse_args()

################################################################################
### DATA SETUP
################################################################################
device = torch.device("cuda:0")

# get path to $SLURM_TMPDIR
slurm_tmpdir = os.getenv('SLURM_TMPDIR')

out_dir = os.path.join(slurm_tmpdir, 'out')
print(f'Files will be saved to: {out_dir}')

set_determinism(seed=0)

# get data files
data_dir = os.path.join(slurm_tmpdir, 'dataset-ISLES22-multimodal-unzipped')
val_list_path = os.path.join(slurm_tmpdir, 'validation_subjs',
                             'single_split', 'validation_subjs_050_001.txt')

print('Loading files...')

# training files (remove validation subjects after)
train_adc = sorted(glob.glob(os.path.join(data_dir, 'rawdata', '*', '*', '*adc.nii.gz')))
train_dwi = sorted(glob.glob(os.path.join(data_dir, 'rawdata', '*', '*', '*dwi.nii.gz')))
#train_flair = sorted(glob.glob(os.path.join(data_dir, 'rawdata', '*', '*', '*flair.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(data_dir, 'derivatives', '*', '*', '*msk.nii.gz')))
train_files = [
    {'image_adc': image_adc, 'image_dwi': image_dwi, 'label': label_name}
    for image_adc, image_dwi, label_name in zip(train_adc, train_dwi, train_labels)
]
print("train_file_sample: ", train_files[0]['image_adc'])
# validation files
val_files = []
with open(val_list_path) as f:
    val_subj_list = f.read().splitlines()
    val_files = [train_file for train_file in train_files if train_file['image_adc'].split('/')[-3] in val_subj_list]
    train_files = [train_file for train_file in train_files if train_file['image_adc'].split('/')[-3] not in val_subj_list]

print(f'Total {len(train_files)} subjects for training.')
print(f'Total {len(val_files)} subjects for validation.')

#train_files_removed = [train_file for train_file in train_files if train_file['image'].split('/')[-4] in missing_list]
#train_files = [train_file for train_file in train_files if train_file['image'].split('/')[-4] not in missing_list]

#val_files_removed = [val_file for val_file in val_files if val_file['image'].split('/')[-4] in missing_list]
#val_files = [val_file for val_file in val_files if val_file['image'].split('/')[-4] not in missing_list]

#print(len(train_files_removed), len(val_files_removed))

#train_files = train_files[:10]
#val_files = val_files[:10]

#print("removing train files: ", train_files_removed)
#print("removing val files: ", val_files_removed)

print(f'Loaded {len(train_files)} subjects for training.')
print(f'Loaded {len(val_files)} subjects for validation.')

################################################################################
### DEFINE TRANSFORMS
################################################################################

# train transforms
# added MaskIntensity implement skull stripping of input image with the specified mask data
# mask data must have the same spatial size as the input image
# all the intensity values of input image corresponding to 0 in the mask data will be set to 0
# others will keep the original value

train_transforms = tf.Compose([
    tf.LoadImaged(keys=['image_adc', 'image_dwi', 'label']),
    tf.AddChanneld(keys=['image_adc', 'image_dwi', 'label']),
    # change all dimensions to be the same
    tf.Spacingd(keys=['image_adc', 'image_dwi', 'label'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
    tf.NormalizeIntensityd(keys=['image_adc', 'image_dwi'], channel_wise=True),
    tf.ResizeWithPadOrCropd(keys=['image_adc', 'image_dwi', 'label'], spatial_size=(250, 250, 150)),
    tf.CopyItemsd(keys=['image_adc', 'image_dwi'], times=1, names=['flipped_adc', 'flipped_dwi']),
    tf.Flipd(keys=['flipped_adc', 'flipped_dwi'], spatial_axis=0),
    tf.ConcatItemsd(keys=['image_adc', 'image_dwi', 'flipped_adc', 'flipped_dwi'], name='image'),
    tf.Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    tf.RandAdjustContrastd(keys=['image'], prob=0.1, gamma=(0.5, 2.5)),
    tf.RandZoomd(keys=['image', 'label'], prob=0.1, min_zoom=0.7, max_zoom=1.1, mode=['area', 'nearest'], keep_size=True),
    tf.RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96,)*3, pos=1, neg=1, num_samples=4),
    tf.ToTensord(keys=['image', 'label']),
    tf.DeleteItemsd(keys=['image_transforms', 'label_transforms'])
])
# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(keys=['image_adc', 'image_dwi', 'label']),
    tf.AddChanneld(keys=['image_adc', 'image_dwi', 'label']),
    # change all dimensions to be the same
    tf.Spacingd(keys=['image_adc', 'image_dwi', 'label'], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
    tf.NormalizeIntensityd(keys=['image_adc', 'image_dwi'], channel_wise=True),
    tf.ResizeWithPadOrCropd(keys=['image_adc', 'image_dwi', 'label'], spatial_size=(250, 250, 150)),
    tf.CopyItemsd(keys=['image_adc', 'image_dwi'], times=1, names=['flipped_adc', 'flipped_dwi']),
    tf.Flipd(keys=['flipped_adc', 'flipped_dwi'], spatial_axis=0),
    tf.ConcatItemsd(keys=['image_adc', 'image_dwi', 'flipped_adc', 'flipped_dwi'], name='image'),
    tf.ToTensord(keys=['image', 'label'])
])

################################################################################
### DATASET AND DATALOADERS
################################################################################

# train dataset
train_ds = CacheDataset(data=train_files, transform=train_transforms,
                        cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)


# valid dataset
val_ds = CacheDataset(data=val_files, transform=val_transforms,
                      cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

################################################################################
### MODEL AND LOSS
################################################################################

model = UNet(
    dimensions=3,
    in_channels=4,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2
).to(device)
loss_function = DiceFocalLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    lambda_focal=0.25
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler() if amp else None

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable model parameters: {total_trainable_params}\n")

################################################################################
### TRAINING LOOP
################################################################################

# general training params
epoch_num = 500  # max epochs 500
early_stop = 125
early_stop_counter = 0
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
val_loss_values = list()
metric_values = list()
epoch_times = list()
total_start = time.time()
post_pred = tf.AsDiscrete(argmax=True, to_onehot=2)
post_label = tf.AsDiscrete(to_onehot=2)

# inference params for patch-based eval
roi_size = (96,)*3
sw_batch_size = 2

print(f'Starting training over max {epoch_num} epochs...')
for epoch in range(epoch_num):
    epoch_start = time.time()
    early_stop_counter += 1
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        inputs, labels= (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
            #batch_data["flipped_image"].to(device)
        )

        #print("input is ", inputs)
        #print("input size is ", inputs.shape)
        #print("flipped input is ", flipped)
        #print("flipped size is ", flipped.shape)
        # input size is  torch.Size([8, 2, 96, 96, 96])
        # flipped size is  torch.Size([8, 1, 197, 233, 189])
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
              f" step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    print(f"time consuming of epoch {epoch + 1} is: {epoch_time:.4f}")

    # validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            step = 0
            metric_sum = 0
            metric_count = 0
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                if amp:
                    with torch.cuda.amp.autocast():
                        # val_output[0] is torch.Size([2, 197, 233, 189])
                        # it has two channels containing the prob for each voxel to correspond to the foreground and background classes (one channel for each)
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        # removing this line causes an error?
                        mask_tensor = mask_tensor.to(val_outputs.device)
                        loss = loss_function(val_outputs, val_labels)
                else:
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    # val_outputs = model(val_inputs)
                    loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()

                dice = compute_meandice(
                    y_pred=post_pred(val_outputs[0]).unsqueeze(0),
                    y=post_label(val_labels[0]).unsqueeze(0),
                    include_background=False,
                ).item()
                metric_count += 1
                metric_sum += dice
            val_loss /= step
            val_loss_values.append(val_loss)
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                print("saved new best metric model")
                early_stop_counter = 0
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
            if early_stop_counter >= early_stop:
                print(f"No validation metric improvement in {early_stop} epochs. "
                      f"Early stopping triggered. Breaking training loop.")
                break
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
      f" total time: {(time.time() - total_start):.4f}")
# save loss and validation metric lists
with open(os.path.join(out_dir, "train_losses.txt"), "wb") as fp:
    pickle.dump(epoch_loss_values, fp)
with open(os.path.join(out_dir, "val_losses.txt"), "wb") as fp:
    pickle.dump(val_loss_values, fp)
with open(os.path.join(out_dir, "val_metrics.txt"), "wb") as fp:
    pickle.dump(metric_values, fp)
with open(os.path.join(out_dir, "epoch_times.txt"), "wb") as fp:
    pickle.dump(epoch_times, fp)

################################################################################
### PLOT TRAINING CURVES
################################################################################

# plot loss and validation metric
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('Epoch Average Loss')
ax[0].set_xlabel('epoch')
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
ax[0].plot(x, y, label='Training loss')
x = [val_interval * (i + 1) for i in range(len(val_loss_values))]
y = val_loss_values
ax[0].plot(x, y, label='Validation loss')
ax[0].legend()
ax[1].set_title('Val Mean Dice')
ax[1].set_xlabel('epoch')
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
ax[1].plot(x, y)
plt.savefig(os.path.join(out_dir, 'training_curves.png'), bbox_inches='tight')

################################################################################
### VALIDATION EVAL AND SAVE PREDS
################################################################################

# evaluate on test set and plot some predictions on axial slices
model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()
df = list()   # collect data for dataframe
cols = ['subject_id', 'dice', 'hausdorff_distance']
with torch.no_grad():
    dice_sum = 0.0
    dice_count = 0
    hd_sum = 0.0
    hd_count = 0
    for i, val_data in enumerate(val_loader):
        subject_id = val_data['label_meta_dict']['filename_or_obj'][0].split('/')[-4]
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        if amp:
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                # val_outputs = model(val_inputs)

        else:
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            # val_outputs = model(val_inputs)

        # added masking for final validation
        #y_hat = post_pred(val_outputs[0])
        #c2 = y_hat[1, ...]
        #masked_output = mask_tensor * c2
        #c1 = torch.logical_not(masked_output, out=torch.empty_like(masked_output))
        #masked_output = torch.stack((c1, masked_output))

        # save prediction
        write_nifti(
            # use masked output
            data=post_pred(val_outputs[0])[1],
            file_name=os.path.join(out_dir, 'preds', f'{subject_id}.nii.gz'),
            affine=val_data['label_meta_dict']['affine'][0]
        )

        # save metrics
        dice = compute_meandice(
            # use masked output
            y_pred=post_pred(val_outputs[0]).unsqueeze(0),
            y=post_label(val_labels[0]).unsqueeze(0),
            include_background=False
        ).item()
        dice_count += 1
        dice_sum += dice
        hd = compute_hausdorff_distance(
            # use masked output
            y_pred=post_pred(val_outputs[0]).unsqueeze(0),
            y=post_label(val_labels[0]).unsqueeze(0),
            include_background=False,
            percentile=95
        ).item()
        hd_count += 1
        hd_sum += hd
        conf_matrix = get_confusion_matrix(
            # use masked output
            y_pred=post_pred(val_outputs[0]).unsqueeze(0),
            y=post_label(val_labels[0]).unsqueeze(0),
            include_background=False
        )
        tp, fp, tn, fn = tuple([item.item() for item in conf_matrix[0][0]])
        row_entry = [subject_id, dice, hd, tp, fp, tn, fn]
        df.append(dict(zip(cols, row_entry)))

