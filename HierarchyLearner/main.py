
import torch
from config import *

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0,"chamfer":100.0,"equillibrium_loss":1.0}

argparser = argparse.ArgumentParser()
# [general config of the training]
argparser.add_argument("--phase",                   default = "0")
argparser.add_argument("--device",                  default = config.device)
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 400 * 3)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--batch_size",              default = 1)
argparser.add_argument("--dataset",                 default = "toy")
argparser.add_argument("--category",                default = ["vase"])
argparser.add_argument("--freeze_perception",       default = False)
argparser.add_argument("--concept_type",            default = False)

# [perception and language grounding training]
argparser.add_argument("--perception",              default = "psgnet")
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 1.00)
argparser.add_argument("--beta",                    default = 1.0)
argparser.add_argument("--loss_weights",            default = weights)

# [additional training details]
argparser.add_argument("--warmup",                  default = True)
argparser.add_argument("--warmup_steps",            default = 300)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [curriculum training details]
argparser.add_argument("--effective_level",         default = 1)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 10,       type=int)
argparser.add_argument("--pretrain_perception",     default = False)

args = argparser.parse_args()

config.perception = args.perception
if args.concept_type: config.concept_type = args.concept_type
args.freeze_perception = bool(args.freeze_perception)
args.lr = float(args.lr)


if args.checkpoint_dir:
    #model = torch.load(args.checkpoint_dir, map_location = config.device)
    model = Halearner(config)
    if "ckpt" in args.checkpoint_dir[-4:]:
        model = torch.load(args.checkpoint_dir, map_location = args.device)
    else: model.load_state_dict(torch.load(args.checkpoint_dir, map_location=args.device))
else:
    print("No checkpoint to load and creating a new model instance")
    model = Halearner(config)
model = model.to(args.device)


if args.pretrain_perception:
    model.load_state_dict(torch.load(args.pretrain_perception, map_location = config.device))


print("using perception: {} knowledge:{} dataset:{}".format(args.perception,config.concept_type,args.dataset))


