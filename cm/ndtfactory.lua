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

function NDTFactory:buildModel(opt)
   local ndt = dp.Sequential()
   -- trunk layer
   ndt:add(
      dp.Neural{
         input_size=opt.feature_size, output_size=opt.expert_width,
         transfer=self:buildTransfer(opt.activation),
         dropout=self:buildDropout(opt.input_dropout and 0.2)
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
      local gater_size = math.ceil(
         opt.gater_width_scales[layer_idx] 
         * opt.gater_width/opt.n_branch^(layer_idx-1)
      )
      print(n_nodes*opt.n_branch..' experts with '
            ..expert_size..' hidden neurons')
      print(n_nodes..' gaters with '..gater_size..' hidden neurons')
      for node_idx = 1,n_nodes do
         local experts = {}
         for expert_idx = 1,opt.n_branch do
            local expert = dp.Neural{
               input_size=input_size, output_size=expert_size,
               transfer=self:buildTransfer(opt.activation),
               dropout=self:buildDropout(opt.expert_dropout and 0.5)
            }
            if layer_idx == opt.n_switch_layer then
               -- last layer of experts is 2-layer MLP
               expert = dp.Sequential{
                  models = {
                     expert,
                     dp.Neural{
                        input_size=expert_size,
                        output_size=#opt.classes,
                        transfer=nn.LogSoftMax(),
                        dropout=self:buildDropout(
                           opt.output_dropout and 0.5
                        )
                     }
                  }
               }
            end
            table.insert(experts, expert)
         end
         --[[ gater ]]--
         local gater = dp.Sequential()
         local g_input_size = input_size
         if opt.gater_dept[layer_idx] == 2 then
            gater:add(
               dp.Neural{
                  input_size=input_size, output_size=gater_size,
                  transfer=self:buildTransfer(opt.activation),
                  dropout=self:buildDropout(opt.gater_dropout and 0.5)
               }
            )
            g_input_size = gater_size
         elseif opt.gater_dept[layer_idx] ~= 1 then
            error"Unsupported gater dept"
         end
         gater:add(
            dp.Equanimous{
               input_size=g_input_size, output_size=opt.n_branch,
               dropout=self:buildDropout(opt.gater_dropout and 0.5),
               transfer=nn.Sigmoid(), n_sample=opt.n_sample,
               n_reinforce=opt.n_reinforce, n_eval=opt.n_eval
            }
         )
         table.insert(nodes, dp.SwitchNode{gater=gater, experts=experts})
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
   return dp.Conditioner{
      criterion = nn.ESSRLCriterion{
         n_reinforce=opt.n_reinforce, n_sample=opt.n_output_sample,
         n_classes=#opt.classes, n_leaf=opt.n_leaf
      },
      visitor = self:buildVisitor(opt),
      feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{
         batch_size=opt.batch_size, sample_type=opt.model_type
      },
      progress = true
   }
end

function NDTFactory:buildValidator(opt)
   return dp.Shampoo{
      criterion = nn.ESSRLCriterion{
         n_reinforce=opt.n_reinforce, n_sample=opt.n_output_sample,
         n_classes=#opt.classes, n_leaf=opt.n_leaf
      },
      feedback = dp.Confusion(),  
      sampler = dp.Sampler{batch_size=1024, sample_type=opt.model_type}
   }
end

function NDTFactory:buildTester(opt)
   return dp.Shampoo{
      criterion = nn.ESSRLCriterion{
         n_reinforce=opt.n_reinforce, n_sample=opt.n_output_sample,
         n_classes=#opt.classes, n_leaf=opt.n_leaf
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
   opt.n_reinforce = math.min(opt.n_reinforce, opt.n_output_sample)
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
      'PGMLPFactory', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGNDTFactory:buildObserver(opt)
   return {
      self._logger,
      dp.PGEarlyStopper{
         start_epoch = 1,
         pg = self._pg,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10, max_error = 70
      },
      dp.PGDone{pg=self._pg}
   }
end
