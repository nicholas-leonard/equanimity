require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL-backed MNIST Neural Decision Tree Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pgndt.lua --collection "MNIST-NDT-2" --hostname "nps" --pid 1 --type cuda --maxTries 100 --maxEpoch 1000 --batchSize 128 --nBranch 8 --learningRate 0.01 --momentum 0.9995 --activation ReLU --gaterWidth 320 --gaterWidthScales "{1,2,2}" --gaterDept "{2,2,2}" --expertWidth 2048 --expertWidthScales "{2,2,2}" --nSwitchLayer 1 --nReinforce 1 --nSample 2 --nEval 1')
cmd:text('Options:')
cmd:option('--nEval', 1, 'number of experts chosen during evaluation')
cmd:option('--nSample', 2, 'number of experts sampled during training')
cmd:option('--nReinforce', 1, 'number of experts reinforced per example during training')
cmd:option('--nBackprop', 1, 'number of experts backpropagated per example during training')
cmd:option('--nBranch', 8, 'number of expert branches per node')
cmd:option('--nSwitchLayer', 1, 'number of switchlayers in the tree')
cmd:option('--learningRate', 0.01, 'learning rate at epoch 0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--weightDecay', 0, 'weight decay factor')
cmd:option('--momentum', 0, 'momentum factor')
cmd:option('--nesterov', false, 'use nesterov momentum')
cmd:option('--expertWidth', 1024, 'width of trunk and combined width of experts times expert width scale')
cmd:option('--expertWidthScales', '{1,1,1}', 'see expertWidth')
cmd:option('--expertLearnScales', '{1,1,1}', 'scales the learning rate')
cmd:option('--gaterWidth', 256, 'width of gater and combinded width of gaters times gater scale factor')
cmd:option('--gaterWidthScales', '{1,2,2}', 'see gaterWidth')
cmd:option('--gaterLearnScales', '{1,1,1}', 'scales the learning rate')
cmd:option('--gaterDept', '{2,2,2}', 'dept of gaters in different layers')
cmd:option('--activation', 'Tanh', 'activation function')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--inputDropout', false, 'apply dropout on inputs, requires "nnx" luarock')
cmd:option('--expertDropout', false, 'apply dropout on experts, requires "nnx" luarock')
cmd:option('--gaterDropout', false, 'apply dropout on gaters, requires "nnx" luarock')
cmd:option('--outputDropout', false, 'apply dropout on outputs, requires "nnx" luarock')
cmd:option('--shareOutput', false, 'share parameters of output layer')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use for this hyperoptimization')
cmd:option('--blockGater', false, 'when true, gater does not backpropagate into previous expert(s)')
cmd:option('--firstDecay', 200, 'epoch at which learning rate is first decayed by a factor of 0.1')
cmd:option('--secondDecay', 400, 'epoch at which learning rate is then decayed by another factor of 0.1')
cmd:option('--collection', 'postgresql-backend hyperoptimization example 2', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host. Only important that each process on same host have different names')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--validRatio', 1/6, 'proportion of train set used for validation')
cmd:option('--epsilon', 0.1, 'probability of sampling from inverse distribution') 
cmd:option('--lambda', 0, 'weight of inverse marginal expert multinomial dist')
cmd:option('--ema', 0.5, 'weight of present for computing exponential moving avg')
cmd:option('--backpropPad', 1, 'dont backpropagate through the backpropPad best experts per example (padding)')
cmd:option('--equanimity', false, 'add a second optimization phase that focuses on experts instead of examples')
cmd:option('--accumulator', 'softmax', 'softmax | normalize')
cmd:option('--trunkLearnScale', 1, 'learning rate scale for the trunk layer')
cmd:option('--gaterGradScale', 1, 'what to multiply gater grad by before adding it to grad sent to previous layer expert or trunk')
cmd:option('--yoshuaBackprop', false, 'use the distribution of example-expert errors to weigh the output gradients.')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--nopg', false, 'dont use postgresql')
cmd:text()
opt = cmd:parse(arg or {})

if opt.useDevice and opt.type == 'cuda' then
   require "cutorch"
   require "nnx"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameters ]]--

local hp = {
   version = 3,
   progress = opt.progress,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = 'mnist',
   random_seed = dp.TimeChoose(),
   model_dept = opt.nSwitchLayer + 2, -- 2 for trunk and leafs
   batch_size = opt.batchSize,
   learning_rate = opt.learningRate,
   learning_decay1 = opt.firstDecay,
   learning_decay2 = opt.secondDecay,
   max_out_norm = opt.maxOutNorm,
   weight_decay = opt.weightDecay,
   momentum = opt.momentum,
   nesterov = opt.nesterov,
   block_gater = opt.blockGater,
   expert_width = opt.expertWidth,
   gater_width = opt.gaterWidth,
   gater_dept = table.fromString(opt.gaterDept),
   trunk_learn_scale = opt.trunkLearnScale,
   expert_width_scales = table.fromString(opt.expertWidthScales),
   expert_learn_scales = table.fromString(opt.expertLearnScales),
   gater_width_scales = table.fromString(opt.gaterWidthScales),
   gater_learn_scales = table.fromString(opt.gaterLearnScales),
   activation = opt.activation,
   input_dropout = opt.inputDropout,
   expert_dropout = opt.expertDropout,
   gater_dropout = opt.gaterDropout,
   output_dropout = opt.outputDropout,
   n_branch = opt.nBranch,
   n_sample = opt.nSample,
   n_eval = opt.nEval,
   n_reinforce = opt.nReinforce,
   n_backprop = opt.nBackprop,
   epsilon = opt.epsilon,
   lambda = opt.lambda,
   ema = opt.ema,
   backprop_pad = opt.backpropPad,
   share_output = opt.shareOutput,
   valid_ratio = opt.validRatio,
   equanimity = opt.equanimity,
   accumulator = opt.accumulator,
   pid = opt.pid,
   hostname = opt.hostname,
   collection = opt.collection,
   gater_grad_scale = opt.gaterGradScale,
   yoshua_backprop = opt.yoshuaBackprop
}

local process_id = opt.hostname .. '.' .. opt.pid

if opt.nopg then
   local logger = dp.FileLogger()
   hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      id_gen=dp.EIDGenerator(process_id),
      hyperparam_sampler = dp.PriorSampler{
         name='NDT+Mnist:dist1', dist=hp --only samples random_seed
      },
      experiment_factory = dp.NDTFactory{
         logger=logger,
         save_strategy=dp.SaveToFile{hostname=opt.hostname}
      },
      datasource_factory=dp.MnistFactory(),
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
   hyperparam_sampler = dp.PriorSampler{
      name='NDT+Mnist:dist1', dist=hp --only samples random_seed
   },
   experiment_factory = dp.PGNDTFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{hostname=opt.hostname, pg=pg}
   },
   datasource_factory=dp.MnistFactory(),
   process_name=process_id,
   logger=logger
}

hyperopt:run()
