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

--[[Load DataSource]]--
datasource = dp.Mnist()

--[[Model]]--
mlp = dp.Sequential()
mlp:add(dp.Linear{input_size=28*28, output_size=50})
mlp:add(dp.Module(nn.Tanh()))
mlp:add(dp.Module(nn.Linear(50, 10)))
mlp:add(dp.Module(nn.Softmax()))

--[[Propagators]]--
train = dp.Optimizer{
   criterion = nn.ClassNLLCriterion(),
   observer = {
      dp.Logger(), 
      dp.LearningRateSchedule{schedule={[30]=0.01, [60]=0.001}}
   },
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=0.9},
      dp.Learn{learning_rate=0.1},
      dp.MaxNorm{max_out_norm=1}
   }
}
valid = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = Confusion(),  
   observer = dp.Logger()
}
test = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = Confusion()
}

--[[Experiment]]--
xp = dp.Experiment{
   optimizer = train,
   validator = valid,
   tester = test,
   observer = dp.EarlyStopper{
      error_report = {'validator','feedback','confusion','accuracy'},
      maximize = true
   }
}

xp:run(datasource)
