------------------------------------------------------------------------
--[[ NCTFactory ]]--
-- An experiment builder using an a Neural Clustering Tree
-- Uses an unsupervised gater
------------------------------------------------------------------------
local NCTFactory, parent = torch.class("dp.NCTFactory", "dp.NDTFactory")
NCTFactory.isNCTFactory = true
   
function NCTFactory:__init(config)
   config = config or {}
   local args, name = xlua.unpack(
      {config},
      'NCTFactory', 'Neural Clustering Tree factory',
      {arg='name', type='string', default='NDT'}
   )
   config.name = name
   parent.__init(self, config)
end

function NCTFactory:buildGater(opt, layer_idx, input_size)
   local gater_size = math.ceil(
      opt.gater_width_scales[layer_idx] 
      * opt.gater_width/opt.n_branch^(layer_idx-1)
   )
   local encode_lrs = opt.encode_learn_scales[layer_idx] 
   local kmeans_lrs = opt.kmeans_learn_scales[layer_idx]
   local gater = dp.Sequential()
   local g_input_size = input_size
   if opt.gater_dept[layer_idx] == 2 then
      gater:add(
         dp.AutoEncoder{
            input_size=input_size, hidden_size=gater_size,
            hidden_transfer=self:buildTransfer(opt.encoding),
            output_transfer=(layer_idx == 1 and self:buildTransfer(opt.activation)) or nn.Sigmoid(),
            input_noise=self:buildDropout(opt.encoder_noise),
            mvstate={learn_scale=encode_lrs}, 
            tags={['gater']=true,['auto-encoder']=true}
         }
      )
      g_input_size = gater_size
   elseif opt.gater_dept[layer_idx] ~= 1 then
      error"Unsupported gater dept"
   end
   gater:add(
      dp.Kmeans{
         input_size=g_input_size, k=opt.n_branch,
         sim_proto=opt.sim_proto, n_sample=opt.n_sample,
         tags={['gater']=true,['kmeans']=true,['no-maxnorm']=true},
         mvstate={learn_scale=kmeans_lrs}
      }
   )
   print('gater with '..gater_size..' hidden neurons')
   return gater
end

function NCTFactory:buildNode(opt, layer_idx, gater, experts)
   return dp.RouterNode{gater=gater, experts=experts}
end

function NCTFactory:buildOptimizer(opt)
   return dp.Conditioner{
      criterion = nn.UGCriterion{
         n_sample=opt.n_output_sample,
         n_classes=#opt.classes, n_leaf=opt.n_leaf,
         n_eval=opt.n_eval, accumulator=opt.accumulator
      },
      visitor = self:buildVisitor(opt),
      feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{
         batch_size=opt.batch_size, sample_type=opt.model_type
      },
      progress = opt.progress
   }
end

function NCTFactory:buildValidator(opt)
   return dp.Shampoo{
      criterion = nn.UGCriterion{
         n_sample=opt.n_output_sample, accumulator=opt.accumulator,
         n_classes=#opt.classes, n_leaf=opt.n_leaf, n_eval=opt.n_eval
      },
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function NCTFactory:buildTester(opt)
   return dp.Shampoo{
      criterion = nn.UGCriterion{
         n_sample=opt.n_output_sample, accumulator=opt.accumulator,
         n_classes=#opt.classes, n_leaf=opt.n_leaf, n_eval=opt.n_eval
      },
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

------------------------------------------------------------------------
--[[ PGNCTFactory ]]--
-- PostgreSQL Neural Decision Tree builder
------------------------------------------------------------------------
local PGNCTFactory, parent = torch.class("dp.PGNCTFactory", "dp.NCTFactory")
PGNCTFactory.isPGNCTFactory = true
   
function PGNCTFactory:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGNCTFactory', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGNCTFactory:buildObserver(opt)
   return {
      self._logger,
      dp.PGEarlyStopper{
         start_epoch = 11,
         pg = self._pg,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10, max_error = opt.max_error
      },
      dp.PGDone{pg=self._pg}
   }
end
