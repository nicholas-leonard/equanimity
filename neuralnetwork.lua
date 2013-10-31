require 'cutorch'
require 'cunn'
torch.setdefaulttensortype('torch.CudaTensor')
require 'dataset/mnist'

--Load datasets:
train_data = dataset.Mnist{which_set='train'}
valid_data = dataset.Mnist{which_set='valid'}
test_data = dataset.Mnist{which_set='test'}

--

