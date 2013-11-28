require 'torch'
require 'sys'
require 'dp'
require 'nn'
require 'paths'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training/Optimization')
cmd:text('Example:')
cmd:text('th neuralnetwork_example.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--numHidden', 200, 'number of hidden units')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:text()
opt = cmd:parse(arg or {})

print(opt)

--[[Expert ID generator]]--
id_gen = dp.EIDGenerator('mypc.pid')

--[[Load DataSource]]--
datasource = dp.Mnist()

--[[Model]]--
mlp = dp.Sequential()
mlp:add(dp.Linear{input_size=datasource._feature_size, output_size=opt.numHidden})
mlp:add(dp.Module(nn.Tanh()))
mlp:add(dp.Linear{input_size=opt.numHidden, output_size=#(datasource._classes)})
mlp:add(dp.Module(nn.SoftMax()))

--[[Propagators]]--
train = dp.Optimizer{
   criterion = nn.ClassNLLCriterion(),
   observer =  {
      dp.Logger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries,
      }
   },
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=opt.momentum},
      dp.Learn{
         learning_rate=opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule={[30]=0.01, [60]=0.001}
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size=opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion(),  
   observer = dp.Logger()
}
test = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion()
}

--[[Experiment]]--
xp = dp.Experiment{
   id_gen = id_gen,
   model = mlp,
   optimizer = train,
   validator = valid,
   tester = test,
   max_epoch = opt.maxEpoch
}

xp:run(datasource)
