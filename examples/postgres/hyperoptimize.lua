require 'dp'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST+MLP Hyperparameter Optimization')
cmd:text('Example:')
cmd:text('$> th hyperoptimize.lua --maxEpoch 10 --maxTries 5')
cmd:text('Options:')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:text()
opt = cmd:parse(arg or {})

--[[ hyperparameter sampling distribution ]]--

dist = {
   max_tries = opt.maxTries,
   max_epoch = opt.maxEpoch,
   model_type = opt.type,
   datasource = 'mnist',
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
      [2] = 0.9, [3] = 0.05, [4] = 10
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
pg = dp.Postgres()

hyperopt = dp.HyperOptimizer{
   collection_name = 'postgresql-backend hyperoptimization example 2',
   id_gen = dp.PGEIDGenerator{pg=pg},
   hyperparam_sampler = dp.PriorSampler{name='MLP+Mnist:dist1', dist=dist},
   experiment_factory = dp.MLPFactory(),
   datasource_factory = dp.MnistFactory()
}

hyperopt:run()
 
