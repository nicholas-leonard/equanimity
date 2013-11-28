require 'torch'
require 'sys'
require 'dp'
require 'nn'
require 'paths'

--[[parse command line arguments]]--
--[[
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-numHidden', 200, 'number of hidden units')
cmd:option('-batchSize', 32, 'number of examples per batch')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:text()
opt = cmd:parse(arg or {})]]--

--[[Expert ID generator]]--
id_gen = dp.EIDGenerator('mypc.pid')

--[[Load DataSource]]--
datasource = dp.Mnist()

--[[Model]]--
mlp = dp.Sequential()
mlp:add(dp.Linear{input_size=28*28, output_size=100})
mlp:add(dp.Module(nn.Tanh()))
mlp:add(dp.Module(nn.Linear(100, 10)))
mlp:add(dp.Module(nn.SoftMax()))

--[[Propagators]]--
train = dp.Optimizer{
   criterion = nn.ClassNLLCriterion(),
   observer =  {
      dp.Logger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = 5,
         start_epoch = 1
      }
   },
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=0.9},
      dp.Learn{
         learning_rate=0.1, 
         observer = dp.LearningRateSchedule{
            schedule={[30]=0.01, [60]=0.001}
         }
      },
      dp.MaxNorm{max_out_norm=1}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size=128},
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
   max_epoch = 10
}

xp:run(datasource)
