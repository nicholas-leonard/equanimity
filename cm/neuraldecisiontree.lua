require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Neural Decision Tree Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuraldecisiontree.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--nEval', 2, 'number of experts chosen during evaluation')
cmd:option('--nSample', 2, 'number of experts sampled during training')
cmd:option('--nReinforce', 1, 'number of experts reinforced during training')
cmd:option('--nExpert', 8, 'number of expert branches in model')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--nExpertHidden', 100, 'number of hidden units per expert')
cmd:option('--nGaterHidden', 200, 'number of hidden units per gater')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--nTrunkHidden', 0, 'if greater than zero, add a trunk dp.Neural layer before dp.SwitchLayer(s)')
cmd:text()
opt = cmd:parse(arg or {})

local target_range = {0.1, 0.9}
print(opt)

--[[Experiment ID generator]]--
id_gen = dp.EIDGenerator('mypc.pid')

--[[Load DataSource]]--
datasource = dp.Mnist()

--[[Model]]--
local input_size = datasource._feature_size
-- neural decision tree
ndt = dp.Sequential()
-- add a trunk layer?
if opt.nTrunkHidden > 0 then
   ndt:add(
      dp.Neural{
         input_size=input_size,
         output_size=opt.nTrunkHidden,
         transfer=nn.Tanh()
      }
   )
   input_size = opt.nTrunkHidden
end

local experts = {}
for i = 1,opt.nExpert do
   table.insert(experts, 
      dp.Neural{
         input_size=input_size,
         output_size=opt.nExpertHidden,
         transfer=nn.Tanh()
      }
   )
end
local gater = dp.Sequential{
   models = {
      dp.Neural{
         input_size=input_size,
         output_size=opt.nGaterHidden,
         transfer=nn.Tanh()
      },
      dp.Equanimous{
         input_size=opt.nGaterHidden,
         output_size=opt.nExpert,
         transfer=nn.Sigmoid(),
         n_test=opt.nTest
      }
   }
}
ndt:add(dp.SwitchNode{gater=gater, experts=experts})

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   ndt:cuda()
end

--[[Propagators]]--
train = dp.Conditioner{
   criterion = nn.ESSRLCriterion{
      n_reinforce=opt.nReinforce, 
      n_sample=opt.nSample,
      n_classes=#(datasource._classes)
   },
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=opt.momentum},
      dp.Learn{
         learning_rate=opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule={[100]=0.01, [200]=0.001}
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size=opt.batchSize, sample_type=opt.type},
   progress = true
}
valid = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{sample_type=opt.type}
}
test = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion(),
   sampler = dp.Sampler{sample_type=opt.type}
}

--[[Experiment]]--
xp = dp.Experiment{
   id_gen = id_gen,
   model = ndt,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      --dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

xp:run(datasource)

