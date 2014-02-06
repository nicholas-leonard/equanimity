require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL-backed Neural Clustering Tree Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pgnct.lua --collection "Mnist-NCT-8-2" --hostname "nps" --pid 1 --type cuda --maxTries 100 --maxEpoch 1000 --batchSize 128 --learningRate 0.1 --momentum 0.9 --encoding Sigmoid --activation Tanh --gaterWidth 10 --gaterWidthScales "{1}" --gaterDept "{2}" --expertWidth 1024 --expertWidthScales "{2}" --excludeMomentum "gater" --nSwitchLayer 1 --nBranch 8 --nSample 2 --nEval 2 --decayPoints "{200,400,1000}" --encoderNoise 0 --encodeLearnScales "{1}" --kmeansLearnScales "{1}" --progress')
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
cmd:option('--gaterWidthScales', '{1,1,1}', 'see gaterWidth')
cmd:option('--encodeLearnScales', '{1,1,1}', 'scales the learning rate of gater auto-encoders')
cmd:option('--kmeansLearnScales', '{1,1,1}', 'scales the learning rate of gater kmeans')
cmd:option('--gaterDept', '{2,2,2}', 'dept of gaters in different layers')
cmd:option('--activation', 'Tanh', 'activation function')
cmd:option('--encoding', 'Tanh', 'activation function used on hidden neurons in auto-encoders')
cmd:option('--encoderNoise', 0, 'dropout probability of [denoising] auto-encoder')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--inputDropout', false, 'apply dropout on inputs, requires "nnx" luarock')
cmd:option('--expertDropout', false, 'apply dropout on experts, requires "nnx" luarock')
cmd:option('--outputDropout', false, 'apply dropout on outputs, requires "nnx" luarock')
cmd:option('--shareOutput', false, 'share parameters of output layer')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use for this hyperoptimization')
cmd:option('--collection', 'postgresql-backend hyperoptimization example 2', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host. Only important that each process on same host have different names')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--validRatio', 1/6, 'proportion of train set used for validation')
cmd:option('--accumulator', 'normalize', 'softmax | normalize')
cmd:option('--trunkLearnScale', 1, 'learning rate scale for the trunk layer')
cmd:option('--outputLearnScale', 1, 'learning rate scale for the output layer')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--nopg', false, 'dont use postgresql')
cmd:option('--datasource', 'Mnist', 'datasource to use : Mnist | NotMnist | Cifar10')
cmd:option('--zca_gcn', false, 'apply GCN followed by ZCA input preprocessing')
cmd:option('--standardize', false, 'apply Standardize input preprocessing')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing to datasource inputs')
cmd:option('--minAccuracy', 0.1, 'minimum accuracy that must be maintained after 10 epochs')
cmd:option('--simProto', 'rev_dist', 'similarity protocol used in k-means gater: rev_dist | uniform')
cmd:option('--excludeMomentum', '', 'comma-separated string of tags to exclude from momentum. example: "gater,output"')
cmd:text()
opt = cmd:parse(arg or {})

if opt.useDevice and opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameters ]]--

local hp = {
   version = 5,
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
   encode_learn_scales = table.fromString(opt.encodeLearnScales),
   kmeans_learn_scales = table.fromString(opt.kmeansLearnScales),
   activation = opt.activation,
   encoding = opt.encoding,
   encoder_noise = opt.encoderNoise,
   input_dropout = opt.inputDropout,
   expert_dropout = opt.expertDropout,
   output_dropout = opt.outputDropout,
   n_branch = opt.nBranch,
   n_sample = opt.nSample,
   n_eval = opt.nEval,
   share_output = opt.shareOutput,
   valid_ratio = opt.validRatio,
   accumulator = opt.accumulator,
   pid = opt.pid,
   hostname = opt.hostname,
   collection = opt.collection,
   zca_gcn = opt.zca_gcn,
   standardize = opt.standardize,
   lecunlcn = opt.lecunLCN,
   max_error = opt.minAccuracy,
   sim_proto = opt.simProto,
   zero_targets = opt.zeroTargets,
   exclude_momentum = _.split(opt.excludeMomentum)
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
      experiment_factory = dp.NCTFactory{
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
   experiment_factory = dp.PGNCTFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{hostname=opt.hostname, pg=pg}
   },
   datasource_factory=dp.ImageClassFactory(),
   process_name=process_id,
   logger=logger
}

hyperopt:run()


