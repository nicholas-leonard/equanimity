require 'dp'
require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST+NDT Hyperparameter Optimization')
cmd:text('Example:')
cmd:text('$> th hyperoptimize.lua --nSwitchLayer 2 --useDevice 1 --type cuda --maxEpoch 500 --maxTries 100 --collection "MnistNDT1" --hostname "myhost.mydomain.com" --pid 1')
cmd:text('Options:')
cmd:option('--collection', 'postgresql-backend hyperoptimization example 2', 'identifies a collection of related experiments')
cmd:option('--hostname', 'localhost', 'hostname for this host')
cmd:option('--pid', 0, 'identifies process on host. Only important that each process on same host have different names')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use for this hyperoptimization')
cmd:option('--nSwitchLayer', 1, 'number of switchlayers in the tree')
cmd:text()
opt = cmd:parse(arg or {})

if opt.useDevice and opt.type == 'cuda' then
   require "cutorch"
   require "nnx"
   cutorch.setDevice(opt.useDevice)
end

--[[ hyperparameter sampling distribution ]]--

local dist = {
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = 'mnist',
   random_seed = dp.TimeChoose(),
   model_dept = opt.nSwitchLayer + 2, -- 2 for trunk and leafs
   batch_size = dp.WeightedChoose{
      [32]=1, [64]=1, [96]=3, [128]=9, [192]=3, [256]=1, [512]=1 
   },
   learning_rate = dp.WeightedChoose{
      [0.05]=1, [0.01]=10, [0.005]=2, [0.001]=1
   },
   learning_decay1 = dp.WeightedChoose{
      ['none']=1, [100]=2, [200]=5, [300]=2, [400]=1
   },
   learning_decay2 = dp.WeightedChoose{
      ['none']=2, [25]=1, [50]=3, [100]=5, [200]=1
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
      [0]=1, [0.8]=1, [0.9]=2, [0.99]=2, [0.995]=10, [0.9995]=2
   },
   nesterov = dp.WeightedChoose{
      [false] = 0.5, [true] = 0.5
   },
   block_gater = dp.WeightedChoose{
      [false] = 10, [true] = 1
   },
   expert_width = dp.WeightedChoose{ --width of trunk
      [256]=1,[512]=5,[768]=10,[1024]=30,[1512]=10,[2056]=5,[4096]=1
   },
   gater_width = dp.WeightedChoose{
      [128]=1, [192]=5, [256]=10, [320]=7, [384]=5, [512]=1
   },
   gater_dept = dp.WeightedChoose{
      [{1,1,1}]=1, [{2,1,1}]=2, [{2,2,2}]=5, [{2,2,1}]=5 
   },
   n_branch = dp.WeightedChoose{
      [6]=1, [7]=2, [8]=10, [9]=2, [10]=1
   },
   expert_width_scales = dp.WeightedChoose{ --excludes trunk
      [{1,1,1}]=5, [{2,2,2}]=20, [{3,3,3}]=5,
      [{2,3,3}]=5, [{3,2,2}]=5, [{2,4,4}]=5
   },
   gater_width_scales = dp.WeightedChoose{ 
      [{1,1,1}]=5, [{1,2,2}]=20, [{1,3,3}]=5,
      [{1,2,3}]=5, [{1,3,2}]=5, [{1,2,4}]=5
   },
   activation = dp.WeightedChoose{
      ['Tanh'] = 7, ['ReLU'] = 5, ['Sigmoid'] = 1
   },
   input_dropout = dp.Choose{false,true},
   expert_dropout = dp.Choose{false,true},
   gater_dropout = dp.Choose{false,true},
   output_dropout = dp.Choose{false,true}
}

if opt.nSwitchLayer >= 2 then
   dist.n_sample = dp.WeightedChoose{[2]=10, [3]=3, [4]=1}
   dist.n_eval = dp.WeightedChoose{[1]=8, [2]=4, [4]=2, [8]=1,}
   dist.n_reinforce = dp.WeightedChoose{[1]=8, [2]=4, [4]=2, [8]=1}
else
   dist.n_sample = dp.WeightedChoose{[2]=10, [3]=3}
   dist.n_eval = dp.WeightedChoose{[1]=10, [2]=2, [3]=1}
   dist.n_reinforce = dp.WeightedChoose{[1]=8, [2]=4, [4]=2, [8]=1}
end

-- you should have postgresql server/client installed
-- A dp database schema should be set up with psql setup.sql
-- The DEEP_PG_CONN environment variable should specify a connection 
-- string like : 
-- dbname='mydatabase'user='username'host='myhost.com'
-- And your /home/username/.pg_pass file should specify your password 
-- for that host, database, username : 
-- myhost.com:5432:mydatabase:username:mypassword
-- You should also chmod go-rwx ~/.pgpass so that other users 
-- are unable to see your password.
-- And then you can securily an easily connect to the database using :
local pg = dp.Postgres()

local process_id = opt.hostname .. '.' .. opt.pid
local logger = dp.PGLogger{pg=pg}

hyperopt = dp.HyperOptimizer{
   collection_name=opt.collection,
   id_gen=dp.PGEIDGenerator{pg=pg},
   hyperparam_sampler = dp.PriorSampler{
      name='NDT+Mnist:dist1', dist=dist
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
