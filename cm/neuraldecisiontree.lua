require 'cm'

-- TODO :
-- hyperparam sampler
-- phase 2
-- table of values
-- graph of values.
-- conv class

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Neural Decision Tree Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuraldecisiontree.lua --batchSize 128 --momentum 0.5 --learningRate 0.01 --nTrunkHidden 400 --nGaterHidden 200 --hiddenScale 2 --type cuda --maxEpoch 1000 --maxTries 100 --nReinforce 1 --nSample 2 --nEval 1')
cmd:text('Options:')
cmd:option('--nEval', 1, 'number of experts chosen during evaluation')
cmd:option('--nSample', 2, 'number of experts sampled during training')
cmd:option('--nReinforce', 1, 'number of experts reinforced during training')
cmd:option('--nBranch', 8, 'number of expert branches per node')
cmd:option('--nSwitchLayer', 1, 'number of switchlayers in the tree')
cmd:option('--hiddenScale', 2, '')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--nBranchHidden', 100, 'number of hidden units per expert')
cmd:option('--nGaterHidden', 200, 'number of hidden units per gater')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--nTrunkHidden', 0, 'if greater than zero, add a trunk dp.Neural layer before dp.SwitchLayer(s)')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use for this hyperoptimization')
cmd:option('--blockGater', false, 'when true, gater does not backpropagate into previous expert(s)')
cmd:option('--epsilon', 0.1, 'probability of sampling from inverse distribution') 
cmd:option('--firstDecay', 200, 'epoch at which learning rate is first decayed by a factor of 0.1')
cmd:option('--secondDecay', 400, 'epoch at which learning rate is then decayed by another factor of 0.1')
cmd:text()
opt = cmd:parse(arg or {})

print(opt)

--[[Experiment ID generator]]--
id_gen = dp.EIDGenerator('mypc.pid')

--[[Load DataSource]]--
datasource = dp.Mnist()
local n_classes = #(datasource._classes)

--[[Model]]--
local input_size = datasource._feature_size
local n_nodes = 1
print(input_size..' pixels')
-- neural decision tree
ndt = dp.Sequential()
-- add a trunk layer?
local hidden_size = opt.nBranchHidden
if opt.nTrunkHidden > 0 then
   ndt:add(
      dp.Neural{
         input_size=input_size, output_size=opt.nTrunkHidden,
         transfer=nn.Tanh()
      }
   )
   input_size = opt.nTrunkHidden
   hidden_size = opt.nTrunkHidden
   print(n_nodes..' trunk with '..input_size..' hidden neurons')
end

for layer_idx = 1,opt.nSwitchLayer do
   n_nodes = opt.nBranch^(layer_idx-1)
   local nodes = {}
   local expert_size = hidden_size/opt.nBranch^layer_idx
   local gater_size = opt.nGaterHidden/opt.nBranch^(layer_idx-1)
   if layer_idx > 1 then
      expert_size = expert_size*opt.hiddenScale
   end
   expert_size = math.ceil(expert_size)
   gater_size = math.ceil(gater_size)
   print(n_nodes*opt.nBranch..' experts with '..expert_size..' hidden neurons')
   print(n_nodes..' gaters with '..gater_size..' hidden neurons')
   for node_idx = 1,n_nodes do
      local experts = {}
      for expert_idx = 1,opt.nBranch do
         local expert = dp.Neural{
            input_size=input_size, output_size=expert_size,
            transfer=nn.Tanh()
         }
         if layer_idx == opt.nSwitchLayer then
            -- last layer of experts is 2-layer MLP
            expert = dp.Sequential{
               models = {
                  expert,
                  dp.Neural{
                     input_size=expert_size, output_size=n_classes,
                     transfer=nn.LogSoftMax()
                  }
               }
            }
         end
         table.insert(experts, expert)
      end
      --[[ gater ]]--
      local gater = dp.Sequential{
         models = {
            dp.Neural{
               input_size=input_size, output_size=gater_size,
               transfer=nn.Tanh()
            },
            dp.Equanimous{
               input_size=gater_size, output_size=opt.nBranch,
               transfer=nn.Sigmoid(), n_sample=opt.nSample,
               n_reinforce=opt.nReinforce, n_eval=opt.nEval,
               epsilon=opt.epsilon
            }
         }
      }
      table.insert(nodes, 
         dp.SwitchNode{
            gater=gater, experts=experts, block_gater=opt.blockGater
         }
      )
   end
   input_size = expert_size
   ndt:add(dp.SwitchLayer{nodes=nodes})
end
print(n_nodes*opt.nBranch..' leafs for '..n_classes..' classes')

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   ndt:cuda()
end

--[[Propagators]]--
local n_output_sample = opt.nSample^opt.nSwitchLayer
local n_leaf = opt.nBranch^opt.nSwitchLayer
train = dp.Conditioner{
   criterion = nn.ESSRLCriterion{
      n_reinforce=opt.nReinforce, n_sample=n_output_sample,
      n_classes=n_classes, n_leaf=n_leaf
   },
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=opt.momentum},
      dp.Learn{
         learning_rate=opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule={
               [opt.firstDecay]=opt.learningRate*0.1, 
               [opt.secondDecay]=opt.learningRate*0.01
            }
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size=opt.batchSize, sample_type=opt.type},
   progress = true
}
valid = dp.Shampoo{
   criterion = nn.ESSRLCriterion{
      n_reinforce=opt.nReinforce, n_sample=n_output_sample,
      n_classes=n_classes, n_leaf=n_leaf, n_eval=opt.nEval
   },
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{sample_type=opt.type}
}
test = dp.Shampoo{
   criterion = nn.ESSRLCriterion{
      n_reinforce=opt.nReinforce, n_sample=n_output_sample,
      n_classes=n_classes, n_leaf=n_leaf, n_eval=opt.nEval
   },
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
         maximize = true, max_epochs = opt.maxTries,
         max_error = 0.75, min_epoch = 10, start_epoch = 11
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

xp:run(datasource)
