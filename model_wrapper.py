import argparse

import PIL
from torchvision import transforms

from dataloaders import custom_transforms as tr
from modeling.deeplab import *


class Wrapper(object):
    def __init__(self, num_class,backbone,resume):
        model = DeepLab(num_classes=num_class,
                        backbone=backbone,
                        sync_bn=False,
                        freeze_bn=False,
                        pretrained=False)

        model = model.cuda()
        if resume:
            checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        self.model = model
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    @staticmethod
    def preprocess(sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCropImg(crop_size=513),
            tr.NormalizeImg(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensorImg()
        ])
        return composed_transforms(sample)

    def predict_file(self, fn):
        img = PIL.Image.open(fn)
        img = self.preprocess(img)
        return self.model(torch.stack([img]).cuda())

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--num-class', type=int, default=2, metavar='N',
                        help='num of classes (default: 2)')
    # checking point
    parser.add_argument('--resume', type=str, default='run/coco/deeplab-resnet/model_best.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'])
    args = parser.parse_args()
    model = Wrapper(args.num_class, args.backbone, args.resume)
    result = model.predict_file("data/coco/train/JPEGImages/1562566287720.jpg")
    import numpy as np
    mask = np.squeeze(result.cpu().detach().numpy())
    mask = mask.transpose([1,2,0])[..., 0]
    output = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
    output.save("output.png")

if __name__ == "__main__":
   main()
