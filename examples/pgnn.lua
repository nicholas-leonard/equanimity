require 'dp'

--[[parse command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pgnn.lua --collection "MnistMLP1" --hostname "myhost.mydomain.com" --pid 1 --batchSize 128 --momentum 0.5')
cmd:text('$> th pgnn.lua --collection "Mnist-MLP-baseline1" --hostname "ub" --pid 1 --batchSize 128 --learningRate 0.1 --momentum 0.995 --modelWidth 1024 --widthScales "{1,0.37109375,0.043945312}" --modelDept 4 --progress')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--firstDecay', 700, 'epoch at which learning rate is first decayed by a factor of 0.1')
cmd:option('--secondDecay', 200, 'number of epochs after first decay when learning rate is to decayed by another factor of 0.1')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--weightDecay', 0, 'weight decay factor')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--nesterov', false, 'use nesterov momentum')
cmd:option('--modelWidth', 1024, 'width of the model in hidden neurons')
cmd:option('--widthScales', '{1,1,1}', 'scales the width of different layers')
cmd:option('--modelDept', 2, 'number of Neural layers (affine transform followed by transfer function) to use')
cmd:option('--activation', 'Tanh', 'activation function')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 200, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropoutProbs', '{0}', 'probability of dropout on inputs to each layer, requires "nnx" luarock')
cmd:option('--datasource', 'Mnist', 'datasource to use : Mnist | NotMnist ')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing to datasource inputs')
cmd:option('--collection', 'hyperoptimization example 1', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host.')
cmd:option('--validRatio', 1/6, 'proportion of train set used for cross-validation')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--nopg', false, 'dont use postgresql')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

local hp = {
   version = 1,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = string.lower(opt.datasource),
   random_seed = dp.TimeChoose(),
   batch_size = opt.batchSize,
   random_seed = dp.TimeChoose(),
   model_dept = opt.modelDept,
   learning_rate = opt.learningRate,
   learning_decay1 = opt.firstDecay,
   learning_decay2 = opt.secondDecay,
   max_out_norm = opt.maxOutNorm,
   weight_decay = opt.weightDecay,
   momentum = opt.momentum,
   nesterov = opt.nesterov,
   model_width = opt.modelWidth,
   width_scales = table.fromString(opt.widthScales),
   activation = opt.activation,
   dropout_probs = table.fromString(opt.dropoutProbs),
   valid_ratio = opt.validRatio,
   pid = opt.pid,
   hostname = opt.hostname,
   collection = opt.collection,
   progress = opt.progress
}

local process_id = opt.hostname .. '.' .. opt.pid

if opt.nopg then
   local logger = dp.FileLogger()
   hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      id_gen=dp.EIDGenerator(process_id),
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name='MLP+'..opt.datasource..':user_dist', dist=hp 
      },
      experiment_factory = dp.MLPFactory{
         logger=logger,
         save_strategy=dp.SaveToFile{hostname=opt.hostname}
      },
      datasource_factory=dp[opt.datasource..'Factory'](),
      process_name=process_id,
      logger=logger
   }
   hyperopt:run()
end

local pg = dp.Postgres()
local logger = dp.PGLogger{pg=pg}

hyperopt = dp.HyperOptimizer{
   collection_name=opt.collection,
   id_gen=dp.PGEIDGenerator{pg=pg},
   hyperparam_sampler = dp.PriorSampler{--only samples random_seed
      name='MLP+'..opt.datasource..':user_dist', dist=hp 
   },
   experiment_factory = dp.PGMLPFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{hostname=opt.hostname, pg=pg}
   },
   datasource_factory=dp[opt.datasource..'Factory'](),
   process_name=process_id,
   logger=logger
}

hyperopt:run()
