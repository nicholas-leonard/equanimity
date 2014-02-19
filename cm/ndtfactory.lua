------------------------------------------------------------------------
--[[ NDTFactory ]]--
-- An experiment builder using an a Neural Decision Tree
------------------------------------------------------------------------
local NDTFactory, parent = torch.class("dp.NDTFactory", "dp.MLPFactory")
NDTFactory.isNDTFactory = true
   
function NDTFactory:__init(config)
   config = config or {}
   local args, name = xlua.unpack(
      {config},
      'NDTFactory', nil,
      {arg='name', type='string', default='NDT'}
   )
   config.name = name
   parent.__init(self, config)
end

function NDTFactory:buildGater(opt, layer_idx, input_size)
   local gater_size = math.ceil(
      opt.gater_width_scales[layer_idx] 
      * opt.gater_width/opt.n_branch^(layer_idx-1)
   )
   local gater_lrs = opt.gater_learn_scales[layer_idx] 
   local gater = dp.Sequential()
   local g_input_size = input_size
   for i = 1,opt.gater_dept[layer_idx]-1 do
      gater:add(
         dp.Neural{
            input_size=input_size, output_size=gater_size,
            transfer=self:buildTransfer(opt.activation),
            dropout=self:buildDropout(opt.gater_dropout and 0.5),
            mvstate={learn_scale=gater_lrs}, tags={['gater']=true}
         }
      )
      g_input_size = gater_size
   end
   print('gater with '..gater_size..' hidden neurons X '..
         opt.gater_dept[layer_idx]..' layers.')
   gater:add(
      dp.Equanimous{
         input_size=g_input_size, output_size=opt.n_branch,
         dropout=self:buildDropout(opt.gater_dropout and 0.5),
         n_sample=opt.n_sample, temperature=opt.temperature,
         temperature_decay=opt.temperature_decay, 
         epsilon_decay=opt.epsilon_decay,
         n_eval=opt.n_eval, epsilon=opt.epsilon, tags={['gater']=true},
         mvstate={learn_scale=gater_lrs}, eval_proto=opt.eval_proto
      }
   )
   return gater
end

function NDTFactory:buildNode(opt, layer_idx, gater, experts)
   return dp.SwitchNode{
      gater=gater, experts=experts, zero_targets=opt.zero_targets,
      gater_grad_scale=opt.gater_grad_scale
   }
end

function NDTFactory:buildModel(opt)
   local ndt = dp.Sequential()
   -- trunk layer
   ndt:add(
      dp.Neural{
         input_size=opt.feature_size, output_size=opt.expert_width,
         transfer=self:buildTransfer(opt.activation),
         dropout=self:buildDropout(opt.input_dropout and 0.2),
         mvstate={learn_scale=opt.trunk_learn_scale}, tags={['trunk']=true}
      }
   )
   print('trunk has '..opt.expert_width..' hidden neurons')
   local input_size = opt.expert_width
   local n_nodes
   for layer_idx = 1,opt.n_switch_layer do
      n_nodes = opt.n_branch^(layer_idx-1) -- 1, 8, 64 ...
      local nodes = {}
      local expert_size = math.ceil(
         opt.expert_width_scales[layer_idx] 
         * opt.expert_width/opt.n_branch^layer_idx
      )
      local expert_lrs = opt.expert_learn_scales[layer_idx] 
      print(n_nodes*opt.n_branch..' experts with '
            ..expert_size..' hidden neurons')
      local shared_output
      for node_idx = 1,n_nodes do
         local experts = {}
         for expert_idx = 1,opt.n_branch do
            local expert = dp.Neural{
               input_size=input_size, output_size=expert_size,
               transfer=self:buildTransfer(opt.activation),
               dropout=self:buildDropout(opt.expert_dropout and 0.5),
               mvstate={learn_scale=expert_lrs}, tags={['expert']=true}
            }
            if layer_idx == opt.n_switch_layer then
               -- last layer of experts is 2-layer MLP
               local output = dp.Neural{
                  input_size=expert_size, output_size=#opt.classes,
                  transfer=nn.SoftMax(), tags={['output']=true},
                  dropout=self:buildDropout(opt.output_dropout and 0.5),
                  mvstate={learn_scale=opt.output_learn_scale}
               }
               -- share output params (convolve output layer on experts)
               if opt.share_output then
                  if shared_output then
                     output:share(shared_output)
                  else
                     shared_output = output
                  end
               end
               expert = dp.Sequential{ models = {expert, output} }
            end
            table.insert(experts, expert)
         end
         local gater = self:buildGater(opt, layer_idx, input_size)
         table.insert(
            nodes, self:buildNode(opt, layer_idx, gater, experts)
         )
      end
      input_size = expert_size
      ndt:add(dp.SwitchLayer{nodes=nodes})
   end
   print(n_nodes*opt.n_branch..' leafs for '..#opt.classes..' classes')
   
   --[[GPU or CPU]]--
   if opt.model_type == 'cuda' then
      require 'cutorch'
      require 'cunn'
      ndt:cuda()
   elseif opt.model_type == 'double' then
      ndt:double()
   elseif opt.model_type == 'float' then
      ndt:float()
   end
   return ndt
end

function NDTFactory:buildOptimizer(opt)
   local optimizer_name = 'Conditioner'
   if opt.equanimity then
      optimizer_name = 'Equanimizer'
   end
   return dp[optimizer_name]{
      criterion = nn.ESSRLCriterion{
         n_sample=opt.n_output_sample,
         n_classes=#opt.classes, n_leaf=opt.n_leaf,
         n_eval=opt.n_eval, accumulator=opt.accumulator,
         sparsity_factor=opt.sparsity_factor, antispec=opt.antispec,
         max_main_class=opt.max_main_class, 
         welfare_factor=opt.welfare_factor
      },
      visitor = self:buildVisitor(opt),
      feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{
         batch_size=opt.batch_size, sample_type=opt.model_type
      },
      progress = opt.progress
   }
end

function NDTFactory:buildValidator(opt)
   return dp.Shampoo{
      criterion = nn.ESSRLCriterion{
         n_sample=opt.n_output_sample, accumulator=opt.accumulator,
         n_classes=#opt.classes, n_leaf=opt.n_leaf, n_eval=opt.n_eval
      },
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function NDTFactory:buildTester(opt)
   return dp.Shampoo{
      criterion = nn.ESSRLCriterion{
         n_sample=opt.n_output_sample, accumulator=opt.accumulator,
         n_classes=#opt.classes, n_leaf=opt.n_leaf, n_eval=opt.n_eval
      },
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function NDTFactory:build(opt, id)
   opt.n_sample = math.min(opt.n_sample, opt.n_branch)
   opt.n_switch_layer = opt.model_dept-2
   opt.n_output_sample = opt.n_sample^opt.n_switch_layer
   opt.n_leaf = opt.n_branch^opt.n_switch_layer
   return parent.build(self, opt, id)
end

------------------------------------------------------------------------
--[[ PGNDTFactory ]]--
-- PostgreSQL Neural Decision Tree builder
------------------------------------------------------------------------
local PGNDTFactory, parent = torch.class("dp.PGNDTFactory", "dp.NDTFactory")
PGNDTFactory.isPGNDTFactory = true
   
function PGNDTFactory:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGNDTFactory', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGNDTFactory:buildObserver(opt)
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
