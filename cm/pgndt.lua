require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL-backed Neural Decision Tree Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pgndt.lua --collection "MNIST-NDT-2" --hostname "nps" --pid 1 --type cuda --maxTries 100 --maxEpoch 1000 --batchSize 128 --nBranch 8 --learningRate 0.01 --momentum 0.9995 --activation ReLU --gaterWidth 320 --gaterWidthScales "{1,2,2}" --gaterDept "{2,2,2}" --expertWidth 2048 --expertWidthScales "{2,2,2}" --nSwitchLayer 1 --nReinforce 1 --nSample 2 --nEval 1')
cmd:text('Options:')
cmd:option('--nEval', 1, 'number of experts chosen during evaluation')
cmd:option('--nSample', 2, 'number of experts sampled during training')
cmd:option('--nBranch', 8, 'number of expert branches per node')
cmd:option('--nSwitchLayer', 1, 'number of switchlayers in the tree')
cmd:option('--learningRate', 0.01, 'learning rate at epoch 0')
cmd:option('--decayPoints', '{400,600,700}', 'epochs at which learning rate is decayed')
cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed at each point')
cmd:option('--linearDecay', false, 'linear decay from first to second from second to third point, etc')
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
cmd:option('--collection', 'postgresql-backend hyperoptimization example 2', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host. Only important that each process on same host have different names')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--validRatio', 1/6, 'proportion of train set used for validation')
cmd:option('--epsilon', 0.1, 'probability of sampling from inverse distribution') 
cmd:option('--epsilonDecay', 0.99, 'epsilon decay')
cmd:option('--temperature', 0.1, 'high temperature makes multinomial distribution more uniform') 
cmd:option('--temperatureDecay', 0.99, 'temperature decay')
cmd:option('--accumulator', 'softmax', 'softmax | normalize')
cmd:option('--trunkLearnScale', 1, 'learning rate scale for the trunk layer')
cmd:option('--outputLearnScale', 1, 'learning rate scale for the output layer')
cmd:option('--gaterGradScale', 1, 'what to multiply gater grad by before adding it to grad sent to previous layer expert or trunk')
cmd:option('--maxMainClass', 0.5, 'maximum proportion of the main class in an expert')
cmd:option('--welfareFactor', 0, 'weight of the constraint on the maximum main class')
cmd:option('--entropyFactor', 0, 'weight of the constraint on the per-expert class entropy')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--nopg', false, 'dont use postgresql')
cmd:option('--datasource', 'Mnist', 'datasource to use : Mnist | NotMnist | Cifar10')
cmd:option('--zca_gcn', false, 'apply GCN followed by ZCA input preprocessing')
cmd:option('--standardize', false, 'apply Standardize input preprocessing')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing to datasource inputs')
cmd:option('--minAccuracy', 0.1, 'minimum accuracy that must be maintained after 10 epochs')
cmd:option('--evalProto', 'MAP', 'how to determine routes from gater activations during evaluation')
cmd:option('--zeroTargets', false, 'zero non-sampled expert-example targets instead using activation as target')
cmd:option('--sparsityFactor', -1, 'increases sparsity of equanimous distribution')
cmd:option('--antispec', false, 'backprop through worst examples in each expert')
cmd:option('--excludeMomentum', '', 'comma-separated string of tags to exclude from momentum. example: "gater,output"')
cmd:text()
opt = cmd:parse(arg or {})

if opt.useDevice and opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameters ]]--

local hp = {
   version = 7,
   progress = opt.progress,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = opt.datasource,
   random_seed = dp.TimeChoose(),
   model_dept = opt.nSwitchLayer + 2, -- 2 for trunk and leafs
   batch_size = opt.batchSize,
   learning_rate = opt.learningRate,
   decay_points = table.fromString(opt.decayPoints),
   decay_factor = opt.decayFactor,
   linear_decay = opt.linearDecay,
   max_out_norm = opt.maxOutNorm,
   weight_decay = opt.weightDecay,
   momentum = opt.momentum,
   nesterov = opt.nesterov,
   expert_width = opt.expertWidth,
   gater_width = opt.gaterWidth,
   gater_dept = table.fromString(opt.gaterDept),
   trunk_learn_scale = opt.trunkLearnScale,
   expert_width_scales = table.fromString(opt.expertWidthScales),
   expert_learn_scales = table.fromString(opt.expertLearnScales),
   output_learn_scale = opt.outputLearnScale,
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
   epsilon = opt.epsilon,
   epsilon_decay = opt.epsilonDecay,
   temperature = opt.temperature,
   temperature_decay = opt.temperatureDecay,
   share_output = opt.shareOutput,
   valid_ratio = opt.validRatio,
   accumulator = opt.accumulator,
   pid = opt.pid,
   hostname = opt.hostname,
   collection = opt.collection,
   gater_grad_scale = opt.gaterGradScale,
   zca_gcn = opt.zca_gcn,
   standardize = opt.standardize,
   lecunlcn = opt.lecunLCN,
   max_error = opt.minAccuracy,
   eval_proto = opt.evalProto,
   zero_targets = opt.zeroTargets,
   sparsity_factor = opt.sparsityFactor,
   antispec = opt.antispec,
   max_main_class = opt.maxMainClass,
   welfare_factor = opt.welfareFactor,
   entropy_factor = opt.entropyFactor,
   exclude_momentum = _.split(opt.excludeMomentum, ',')
}

local process_id = opt.hostname .. '.' .. opt.pid

if opt.nopg then
   local logger = dp.FileLogger()
   hyperopt = dp.HyperOptimizer{
      collection_name=opt.collection,
      id_gen=dp.EIDGenerator(process_id),
      hyperparam_sampler = dp.PriorSampler{--only samples random_seed
         name='NDT+'..opt.datasource..':user_dist', dist=hp 
      },
      experiment_factory = dp.NDTFactory{
         logger=logger,
         save_strategy=dp.SaveToFile{hostname=opt.hostname}
      },
      datasource_factory=dp.ImageClassFactory(),
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
      name='NDT+'..opt.datasource..':user_dist', dist=hp 
   },
   experiment_factory = dp.PGNDTFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{hostname=opt.hostname, pg=pg}
   },
   datasource_factory=dp.ImageClassFactory(),
   process_name=process_id,
   logger=logger
}

hyperopt:run()


