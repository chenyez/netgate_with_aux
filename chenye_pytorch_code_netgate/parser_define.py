
import argparse

def train_args_define():

    parser = argparse.ArgumentParser(description='pytorch adv train for CIFAR10')
    
    parser.add_argument('--model-save-name',
                        type=str,
                        default='normal_save',
                        help='save name for c urrent usage model')
    
    parser.add_argument('--train-batch-size', 
                        type=int, 
                        default=16, 
                        metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', 
                        type=int, 
                        default=1000, 
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=5, 
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--seed', 
                        type=int, 
                        default=1, 
                        metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device number')


    args = parser.parse_args()

    return args


def parser_print(args):
    print('\n parser arguments:')
    for arg in vars(args):
        print ('\t', arg, getattr(args, arg))
