from basicsr.data.paired_image_dataset import PairedImageDataset
from omegaconf import OmegaConf

opt = OmegaConf.load('options/train/underwater/swin_UEIB.yml')
dataset = PairedImageDataset(opt['datasets']['val'])
print("VAL images:", len(dataset))
