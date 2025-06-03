import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# checkpoint = torch.load('./runs/BddDataset/checkpoint.pth')
# checkpoint = torch.load('./runs/BddDataset/_2025-05-22-15-25/epoch-300.pth')
# checkpoint = torch.load('./runs/BddDataset/_2025-05-22-15-25/final_state.pth')
checkpoint = torch.load('/home/huayi/hhy/YOLOP/runs/RobotViewDataset/_2025-05-26-21-15/epoch-602.pth')
print(checkpoint.keys())
optimizer = checkpoint['optimizer']
# print(checkpoint['optimizer'])

# load img
# import cv2
#
# img = cv2.imread("/home/huayi/hhy/YOLOP/Datasets/lane_robot_2/gt_instance_image/val/0009.png", -1)
#
# print(img.shape)
