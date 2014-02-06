require 'en'

--[[parse command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('PostgreSQL Enhancer Training/Optimization')
cmd:text('Example:')
cmd:text('$> th pgen.lua --collection "Mnist-En-1" --hostname "myhost.mydomain.com" --pid 1 --batchSize 128 --momentum 0.5')
cmd:text('$> th pgen.lua --collection "Mnist-En-baseline1" --hostname "ub" --pid 1 --batchSize 128 --learningRate 0.1 --momentum 0.995 --modelWidth 1024 --layerScales "{1,1,1}" --modelDept 3 --progress')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoints', '{400,600,700}', 'epochs at which learning rate is decayed')
cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed at each point')
cmd:option('--linearDecay', false, 'linear decay from first to second from second to third point, etc')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--weightDecay', 0, 'weight decay factor')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--nesterov', false, 'use nesterov momentum')
cmd:option('--modelWidth', 1024, 'width of the model in hidden neurons')
cmd:option('--layerScales', '{1,1,1}', 'scales the width of different layers')
cmd:option('--encoderScales', '{0.2,0.2,0.2}', 'scales the width of different layer encoders')
cmd:option('--modelDept', 3, 'number of Neural layers (affine transform followed by transfer function) to use')
cmd:option('--activation', 'Tanh', 'layer activation function')
cmd:option('--encoding', 'Sigmoid', 'encoder activation function')
cmd:option('--evalProto', 'layer-only', 'evaluation protocol : layer-only | average | product')
cmd:option('--updateScales', '{0.1,0.1,0.1}', 'learning rate for substracting act gradient from act')
cmd:option('--mixtureCoeffs', '{0.5,0.5,0.5}', 'Weight of layer activations. does not apply to evalProto=layer-only')
cmd:option('--bpaeCoeffs', '{0,0,0}', 'backpropagate auto-encoder coeff*gradients into layer')
cmd:option('--inputNoises', '{0,0,0}', 'dropout prob for noise into encoder')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 200, 'maximum number of epochs to try to find a better local minima for early-stopping')
--cmd:option('--dropoutProbs', '{0}', 'probability of dropout on inputs to each layer, requires "nnx" luarock')
cmd:option('--datasource', 'Mnist', 'datasource to use : Mnist | NotMnist | Cifar10')
cmd:option('--zca_gcn', false, 'apply GCN followed by ZCA input preprocessing')
cmd:option('--standardize', false, 'apply Standardize input preprocessing')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing to datasource inputs')
cmd:option('--collection', 'hyperoptimization example 1', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host.')
cmd:option('--validRatio', 1/6, 'proportion of train set used for cross-validation')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--nopg', false, 'dont use postgresql')
cmd:option('--minAccuracy', 0.1, 'minimum accuracy that must be maintained after 10 epochs')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
   require "cutorch"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

local hp = {
   version = 2,
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = opt.datasource,
   random_seed = dp.TimeChoose(),
   batch_size = opt.batchSize,
   model_dept = opt.modelDept,
   learning_rate = opt.learningRate,
   decay_points = table.fromString(opt.decayPoints),
   decay_factor = opt.decayFactor,
   linear_decay = opt.linearDecay,
   max_out_norm = opt.maxOutNorm,
   weight_decay = opt.weightDecay,
   momentum = opt.momentum,
   nesterov = opt.nesterov,
   model_width = opt.modelWidth,
   layer_scales = table.fromString(opt.layerScales),
   encoder_scales = table.fromString(opt.encoderScales),
   activation = opt.activation,
   encoding = opt.encoding,
   eval_proto = opt.evalProto,
   update_scales = table.fromString(opt.updateScales),
   mixture_coeffs = table.fromString(opt.mixtureCoeffs),
   input_noises = table.fromString(opt.inputNoises),
   bpae_coeffs = table.fromString(opt.bpaeCoeffs),
   --dropout_probs = table.fromString(opt.dropoutProbs),
   valid_ratio = opt.validRatio,
   pid = opt.pid,
   hostname = opt.hostname,
   collection = opt.collection,
   progress = opt.progress,
   zca_gcn = opt.zca_gcn,
   standardize = opt.standardize,
   lecunlcn = opt.lecunLCN,
   max_error = opt.minAccuracy
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
      experiment_factory = dp.EnFactory{
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
      name='En+'..opt.datasource..':user_dist', dist=hp 
   },
   experiment_factory = dp.PGEnFactory{
      logger=logger, pg=pg, 
      save_strategy=dp.PGSaveToFile{hostname=opt.hostname, pg=pg}
   },
   datasource_factory=dp.ImageClassFactory(),
   process_name=process_id,
   logger=logger
}

hyperopt:run()
