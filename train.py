import os
from tool.config import Config
from modules.detection import get_model, get_loss, Trainer, get_dataloader, get_metric
import argparse

# required arg

parser = argparse.ArgumentParser()
parser.add_argument('--continue_train', required=True, type=bool)
continue_train = vars(parser.parse_args())['continue_train']

config = Config("./tool/config/detection/configs.yaml")
class Arguments:
    print_per_iter = 10
    val_interval = 1
    save_interval = 50
    resume = None
    saved_path = './weights'
    freeze_backbone = True
args = Arguments
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices


trainloader, valloader = get_dataloader(config)

criterion = get_loss(config.loss).cuda()
metric = get_metric(config)

model = get_model(config.model)

trainer = Trainer(args=args,
                  config=config,
                  model=model,
                  criterion=criterion,
                  metric=metric,
                  train_loader=trainloader,
                  val_loader=valloader)
if continue_train:
    trainer._load_checkpoint('weights/PANNet_last.pth', resume = True)
trainer.train()