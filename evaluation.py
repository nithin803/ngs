from train import evaluate
from utils import *
from dataset import *
from NN_AOG import NNAOG
from diagnosis import ExprTree

import torch
import numpy as np

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--mode', default='BS', type=str, help='choose mode. BS or RL or MAPO' )
parser.add_argument('--nstep', default=5, type=int, help='number of steps of backsearching')
parser.add_argument('--pretrain', default=None, type=str, help='pretrained symbol net')
# Dataloader
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--batch_size', default=64, type=int)
# seed
parser.add_argument('--random_seed', default=123, type=int, help="numpy random seed")
parser.add_argument('--manual_seed', default=17, type=int, help="torch manual seed")
# Run
parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
parser.add_argument('--decay', default=0.99, type=float, help="reward decay")
parser.add_argument('--num_epochs', default=5, type=int, help="number of epochs")
parser.add_argument('--n_epochs_per_eval', default=1, type=int, help="test every n epochs")
parser.add_argument('--output_dir', default='output', type=str, help="output directory")

# train or eval moed
parser.add_argument('--model_mode', default='eval', type=str, help='Choose model_mode. train or eval')
parser.add_argument('--model_path', default='output/trained_model.ckpt', type=str, help='path to saved Check Point output/trained_model.ckpt')


def eval_model(opt):
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.manual_seed)
    train_set = MathExprDataset('train', numSamples=int(10000*opt.data_used), randomSeed=777)
    test_set = MathExprDataset('test')
    print('train:', len(train_set), '  test:', len(test_set))
    model = NNAOG().to(device)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    mode = opt.mode
    nstep = opt.nstep
    num_workers = opt.num_workers
    batch_size = opt.batch_size
    lr = opt.lr
    reward_decay = opt.decay
    num_epochs = opt.num_epochs
    n_epochs_per_eval = opt.n_epochs_per_eval
    buffer_weight = 0.5

    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=lr)

    reward_moving_average = None

    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers, collate_fn=MathExpr_collate)
    acc, sym_acc = evaluate(model, eval_dataloader)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100 * acc, 100 * sym_acc))

opt = parser.parse_args()
print(opt)
if opt.model_mode == 'eval':
    eval_model(opt)


