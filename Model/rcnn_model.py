import torchvision
import torch
import torch.utils
import sys
import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict, deque
import math
import copy
import pandas as pd

sys.path.insert(1, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\ThesisVQA\Dataset')
sys.path.insert(2, r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\utils\vision\references\detection')

# from engine import train_one_epoch, evaluate


from vision_dataset import VisionDataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import v2 as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_faster_rcnn_mobilenet(num_classes):

    # Load a pre-trained MobileNet V2 model
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    # Get the features part of MobileNet V2
    # backbone = mobilenet.features
    # FasterRCNN needs to know the number of output channels in a backbone. For MobileNet V2, it's 1280
    backbone= mobilenet.features
    backbone.out_channels = 1280

    # Let’s create an AnchorGenerator for the FPN which by default has 5 feature maps
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # RoI pooler is used to pool ROIs to a fixed size, so that they can be fed to the classifier and regressor
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Put the pieces together inside a FasterRCNN model.
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 8)


# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# num_classes=3

# in_features = model.roi_heads.box_predictor.cls_score.in_features

# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features

# backbone.out_channels = 1280

# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),),
#     aspect_ratios=((0.5, 1.0, 2.0),)
# )

# roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#     featmap_names=['0'],
#     output_size=7,
#     sampling_ratio=2
# )

# model = FasterRCNN(
#     backbone,
#     num_classes,
#     rpn_anchor_generator=anchor_generator,
#     box_roi_pool=roi_pooler
# )

# model = get_faster_rcnn_mobilenet(5)

def get_transform(train):
    transforms = []
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=False))
    # transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def collate_fn(batch): 
     return tuple(zip(*batch)) 

def optional_output():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 8)
    dataset = RCNNDataset('..\VQA_New_Dataset_Aerial.v1i.coco', 'train', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
        )

    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)

    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print(predictions[0])

optional_output()


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 5
# use our dataset and defined transformations
dataset = RCNNDataset('..\Aquarium', 'train', get_transform(train=True))
dataset_test = RCNNDataset('..\Aquarium', 'valid', get_transform(train=False))







# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# get the model using our helper function

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005
)

# # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1
# )

# let's train it just for 2 epochs
num_epochs = 20

# for epoch in range(num_epochs):
    
#     # train for one epoch, printing every 10 iterations
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#     print("DONE TRAINING PART!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # update the learning rate
#     # lr_scheduler.step()
#     # evaluate on the test dataset
#     evaluate(model, data_loader_test, device=device)
#     print("DONE EVALUATION HERES EVALUATION DATA ")


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        print(targets)
        print("Next Target")
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))




for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch)

print("That's it!")


import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from torchvision.io import read_image

image = read_image(r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\VQA\ThermalCheetah\test\IMG_0028_jpeg.rf.514f114438ecc7b4598469edfa2ab9d5.jpg')
eval_transform = get_transform(train=False)

torch.save(model.state_dict(), r'C:\Users\Nathon Rayon\.conda\envs\VQATorch\VQA\model\model_new.pt')

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")



plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))