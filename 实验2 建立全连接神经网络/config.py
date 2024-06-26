import argparse
parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA if available')

# data in/out and dataset
parser.add_argument('--dataset_path',default = r'./data/',help='fixed trainset root path')

parser.add_argument('--save',default=r'./model.pth', help='save path of trained model')

parser.add_argument('--predict',default=r'./model.pth',help='save path of predict model')

parser.add_argument('--batch_size', type=list, default=100,help='batch size of trainset')

# Model parameters
parser.add_argument('--input_size', type=int, default=784, help='Input size of the model (default: 784)')

parser.add_argument('--hidden_size', type=int, default=256, help='Size of hidden layers (default: 256)')

parser.add_argument('--output_size', type=int, default=10, help='Output size of the model (default: 10)')

# train
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.99, metavar='M',help='SGD momentum (default: 0.5)')

parser.add_argument('--weight_decay', type=float, default=3e-5, metavar='W',help='SGD weight_decay (default: 3e-5)')

parser.add_argument('--nesterov', type=bool, default=True, help='SGD nesterov (default: True)')

parser.add_argument('--early-stop', default=20, type=int, help='early stopping (default: 20)')
#args = parser.parse_args()
args, unknown = parser.parse_known_args()
