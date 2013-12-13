require 'dp'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST+MLP Hyperparameter Optimization')
cmd:text('Example:')
cmd:text('$> th hyperoptimize.lua --maxEpoch 500 --maxTries 50 --collection "MnistMLP1" --hostname "myhost.mydomain.com" --pid 1')
cmd:text('Options:')
cmd:option('--collection', 'hyperoptimization example 1', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host. Only important that each process on same host have different names')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--useDevice', nil, 'sets the device (GPU) to use for this hyperoptimization')
cmd:text()
opt = cmd:parse(arg or {})

if opt.useDevice then
   opt.type = 'cuda'
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

dist = {
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = 'mnist',
   batch_size = dp.WeightedChoose{
      [32]=10, [64]=7, [128]=5, [256]=4, [16]=3, [8]=2, [512]=1 
   },
   learning_rate = dp.WeightedChoose{
      [0.5]=0.1, [0.1]=0.8, [0.05]=0.1, [0.01]=0.3, [0.001]=0.1
   },
   learning_schedule = dp.WeightedChoose{
      ['100=/10,200=/10']=0.1, ['100=/10,150=/10']=0.1,
      ['200=/10,250=/10']=0.1, ['none']=0.3
   },
   weight_constraint = dp.WeightedChoose{
      ['weightdecay']=0.2, ['maxnorm']=0.7, ['both']=0.1
   },
   max_out_norm = dp.WeightedChoose{
      [0.5] = 0.1, [1] = 0.7, [2] = 0.2
   },
   weight_decay = dp.WeightedChoose{
      [0.0005] = 0.1, [0.00005] = 0.7, [0.000005] = 0.2
   },
   momentum = dp.WeightedChoose{
      [0] = 1, [0.5] = 0.1, [0.7] = 0.1, [0.9] = 0.3, [0.99] = 0.5
   },
   nesterov = dp.WeightedChoose{
      [false] = 0.5, [true] = 0.5
   },
   model_dept = dp.WeightedChoose{
      [2] = 0.9, [3] = 0.05, [4] = 0.05
   },
   model_width = dp.WeightedChoose{
      [128]=0.1, [256]=0.2, [512]=0.3, [1024]=0.3, [2048]=0.1
   },
   width_scales = dp.WeightedChoose{
      [{1,1,1}]=0.5,      [{1,0.5,0.5}]=0.1, [{1,1,0.5}]=0.1,
      [{1,0.5,0.25}]=0.1, [{0.5,1,0.5}]=0.1, [{1,0.25,0.25}]=0.1
   },
   activation = dp.WeightedChoose{
      ['Tanh'] = 0.4, ['ReLU'] = 0.5, ['Sigmoid'] = 0.1
   }
}

local process_id = opt.hostname .. '.' .. opt.pid

local logger = dp.FileLogger()
hyperopt = dp.HyperOptimizer{
   collection_name = opt.collection,
   id_gen = dp.EIDGenerator(process_id),
   hyperparam_sampler = dp.PriorSampler{name='MLP+Mnist:dist1', dist=dist},
   experiment_factory = dp.MLPFactory{logger=logger},
   datasource_factory = dp.MnistFactory(),
   process_name = process_id,
   logger = logger
}

hyperopt:run()
 
