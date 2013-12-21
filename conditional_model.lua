------------------------------------------------------------------------
--[[ ESSRLCriterion ]]--
-- Equanimous Sparse Supervised Reinforcement Learning
-- takes a table of tables as input.
-- TODO : build as container or model?
------------------------------------------------------------------------
local ESSRLCriterion, parent = torch.class("nn.ESSRLCriterion")

function ESSRLCriterion:__init(config)
   config = config or {}
   local args, criterion, n_test = xlua.unpack(
      {config},
      'ESSRLCriterion', nil,
      {arg='criterion', type='nn.Criterion', default=nn.ClassNLLCriterion,
       help='Criterion to be used for optimizing winning experts'},
      {arg='n_test', type='number', default=1,
       help='number of experts to use during evaluation (test time)'}
   )
   self._criterion = criterion
   self._n_test = n_test
end

function ESSRLCriterion:forward(inputs, targets, indices)
   assert(type(inputs) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local examples = {}
   for i = 1,indices:size()[1] do
      examples[indices[i]] = {
         target = targets[i],
         experts = {},
         id = indices[i]
      }
   end
   -- example expert pair : eepair
   local example, eepair, criterion
   local err = 0
   -- The last layer of the tree are its leafs,
   -- where each represents an expert.
   -- Measure error for each example-expert pair
   for leaf_id,leaf_state in pairs(inputs) do
      -- convert to double (from cuda)
      leaf_state.act = leaf_state.act:double()
      for i = 1, leaf_state.indices:size()[1] do
         example = examples[leaf_state.indices[i]] 
         criterion = self._criterion()
         eepair = {
            id = leaf_id,
            act = leaf_state.act[i,:]),
            criterion = criterion,
            err = criterion.forward(eepair.act, example.target)
            alpha = leaf_state.alpha[i]
         }
         example.full_err = example.full_err + eepair.err
         err = err + eepair.err
         example.experts[leaf_id] = eepair
      end
   end
   self._examples = examples
   self._full_err = err/#inputs
   err = 0
   -- For each example, reinforce winning experts (those that have 
   -- least error) by allowing them to learn and backward-propagating
   -- the indices of winning experts for use by the gaters which will
   -- reinforce the winners.
   local experts, winners, sum, wmean_err
   for example_idx, example in pairs(examples) do
      -- orders experts by ascending error
      experts = _.sortBy(
         _.values(examples.experts), 
         function(expert) return expert.err end
      )
      -- keep n_test winners
      winners = _.slice(experts, 1, n_test)
      example.winners = winners
      example.losers = _.slize(experts, n_test+1, #experts)
      -- error measures of winners
      sum = 0; 
      wmean_act = winners[1].act:clone():zero()
      for i,expert in ipairs(winners) do
         wmean_act:add(expert.alpha, expert.act)
         sum = sum + expert.alpha
      end
      -- error of weighted mean of expert activations (true error)
      wmean_act:div(sum)
      example.err = self._criterion():forward(wmean_act, example.target)
      err = err + example.err
   end
   self._err = err/#inputs
   return err, self._full_err
end

function ESSRLCriterion:backward(inputs, targets, indices)
   for input_id, input_state in pairs(inputs) do
      input_state.reinforce = {}
      input_state.grad
   end
   -- compute gradients for the winning example-expert pairs
   for id, example in pairs(self._examples) do
      for i, winner in ipairs(example.winners) do
         table.insert(inputs[winner.id].reinforce, winner.id)
      end
   end
end

function ESSRLCriterion:evaluate(inputs, targets, indices)
   
end


function ESSRLCriterion:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function ESSRLCriterion:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   return self
end

function ESSRLCriterion:accumulatedOutputs()
   for id, example in pairs(self._examples) do
      
   end
end


------------------------------------------------------------------------
--[[ Equanimous ]]--
-- samples routes from multinomial
-- reinforces winning routes using MSE with 0/1 targets
-- epsilon-greedy exploration
-- affine transform followed by non-linearity (transfer function)
-- uses sigmoid transfer function by default for inputs to multinomial
------------------------------------------------------------------------
local Equanimous, parent = torch.class("dp.Equanimous", "dp.Neural")

--E-greedy : http://junedmunshi.wordpress.com/tag/e-greedy-policy/
function Equanimous:__init(config)
   config = config or {}
   local args, n_train, n_test, targets, transfer, epsilon 
      = xlua.unpack(
      {config},
      'Equanimous', nil,
      {arg='n_train', type='number', 
       help='number of experts sampled per example during training.'},
      {arg='n_test', type='number', 
       help='number of experts sampled per example during testing.'},
      {arg='targets', type='table', default={0.1,0.9},
       help='targets used to diminish (first) and reinforce (last)'},
      {arg='transfer', type='nn.Module', default=nn.Sigmoid(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='epsilon', type='number', default=0.1,
       help='probability of sampling from uniform instead of multinomial'}
   )
   config.typename = config.typename or 'equanimous'
   config.transfer = transfer
   parent.__init(self, config)   
   self._n_train = n_train
   self._n_test = n_test
   self._targets = targets
   self._epsilon = epsilon
end

function Equanimous:setup(config)
   parent.setup(self, config)
end   

function Equanimous:_forward(gstate)
   -- affine transform + transfer function
   parent._forward(gstate)
   -- sample n_train samples from a multinomial without replacement
   local n_train = gstate.n_train or self._n_train
   self.ostate.routes 
      = dp.multinomial_without_replacement(self.ostate.act, n_train)
end

function Equanimous:_backward(gstate, scale)
   local reinforce = self.ostate.reinforce
   
end

function Equanimous:_evaluate(gstate)
   if self._evaluate_protocol = 'MAP' then
      self._evaluateMap(gstate)
   end
end

function Equanimous:report()

end

function Equanimous:_evaluateMAP(gstate)
   
end


------------------------------------------------------------------------
--[[ SwitchNode ]]--
-- a switched node in a tree.
-- branches are experts.
-- each example gets routed through different branches by the gater.
------------------------------------------------------------------------
local SwitchNode, parent = torch.class("dp.SwitchNode", "dp.Container")
SwitchNode.isSwitchNode = true

function SwitchNode:__init(config)
   config = config or {}
   local args, gater = xlua.unpack(
      {config},
      'SwitchNode', nil,
      {arg='gater', type='dp.Model'},
      {arg='experts', type='table'}
   )
   config.typename = 'switchnode'
   parent.__init(self, config)
   self._gater = gater
   self._experts = experts
   self._models = _.concat(self._gater, self._experts)
end


function SwitchNode:setup(config)
   parent.setup(self, config)
   config.container = self
   config.predecessor = self._predecessor
   config.successor = self._successor
   for i, expert in ipairs(self._experts) do
      config.id = self:id():create('e'..i)
      expert:setup(config)
   end
   self._gater:setup(config)
   self._data_view = self._experts[1]:dataView()
end

function SwitchNode:report()
   local report = {
      typename=self._typename, 
      num_experts=#self._experts,
      gater = self._gater:report()
   }
   for i, expert in ipairs(self._experts) do 
      report[i] = expert:report()
   end
   return report
end

function SwitchNode:size()
   return #self._experts
end

function SwitchNode:get(index)
   return self._experts[index]
end

function SwitchNode:_forward(gstate)
   self.estate = {}
   -- forward gater to get routes
   self._gater:forward(gstate)
   -- routes is a matrix of indices
   local routes = self._gater.ostate.routes
   local input_act = self.istate.act
   local input_indices = self.istate.indices
   local batch_indices = {}
   local expert_examples = {}
   -- accumulate example indices for each experts
   for example_idx = 1,routes:size()[1] do
      for sample_idx = 1,routes:size()[2] do
         local expert_idx = routes[{example_idx,sample_idx}]
         local examples = expert_examples[expert_idx] or {}
         table.insert(examples, example_idx)
         expert_examples[expert_idx] = examples
      end
   end
   local expert_batches = {}
   -- create a batch examples for each expert
   for expert_idx, examples in pairs(expert_examples) do
      local indices = torch.LongTensor(examples)
      expert_batch = {
         batch_indices = indices,
         act = input_act:index(1, indices),
         indices = input_indices:index(1, indices)
      }      
      expert_batches[expert_idx] = examples
   end
   -- 
   
   for i = 1,#self._experts do
      -- column in routes is associated to an expert
      batch_indices[i] = routes:sub(2, i)
      -- get examples to be propagated to expert
      self.estate[i] = input_act:index(1, batch_indices[i])
      -- propagate batch to expert
      self._experts[i]:
   end
   -- Will need this for backward
   self._state.routing_table = routing_table
   for i=1,#self._models do 
      self._models[i]:forward(gstate)
   end 
end

function SwithNode:_evaluate(gstate)
   local routes = self._gater:evaluate(gstate)
   
end

function SwitchNode:_backward(gstate, scale)
   scale = scale or 1
   for i=#self._models,1,-1 do
      self._models[i]:backward(scale)
   end
end

function SwitchNode:_accept(visitor)
   for i=1,#self._models do 
      self._models[i]:accept(visitor)
   end 
   visitor:visitContainer(self)
end

function SwitchNode:zeroGradParameters()
  for i=1,#self._models do
     self._models[i]:zeroGradParameters()
  end
end

function SwitchNode:_update(gstate)
   for i=1,#self._models do
      self._models[i]:update(gstate)
   end
end

function SwitchNode:reset(stdv)
   for i=1,#self._models do
      self._models[i]:reset(stdv)
   end
end

function SwitchNode:parameters()
   error"NotImplementedError"
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   for i=1,#self._models do
      local params = self._models[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function SwitchNode:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'dp.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self._models do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self._models do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self._models[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end

------------------------------------------------------------------------
--[[ SwitchLayer ]]--
-- during evaluation, samples from a hierarchical multinomial, where 
-- the probability of each multinomial is taken into account, and where 
-- samples are drawn across all switch-nodes in the layer. 
------------------------------------------------------------------------
local SwitchLayer, parent = torch.class("dp.SwitchLayer", "dp.ParallelTable")

function SwitchLayer:__init(config)
   
end

function SwitchLayer:_evaluate(gstate)

end

function SwitchLayer:_backward(gstate, scale)

end

------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Optimizer")

function Conditioner:__init()
   
end

      
function Conditioner:propagateBatch(batch)   
   local model = self._model
   --[[ Phase 1 : Focus on examples ]]--
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local gstate = {focus='examples'}
   model.istate.act = batch:inputs()
   model.istate.indices = batch:indices()
   model:forward()
   
   -- average loss (a scalar)
   batch:setLoss(
      self._criterion:forward(model.ostate, batch:targets(), batch:indices())
   )
   batch:setOutputs(self._criterion:accumulateOutputs())
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneFeedback", 
                          self:report(), batch)
   
   --[[ backpropagate ]]--
   self._criterion:backward(model.ostate, batch:targets(), batch:indices())
   model:backward()

   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          self:report(), batch)
                          
   --[[ Phase 2 : Focus on experts ]]--
   -- sample a batch of experts for phase 2
   local experts = self._expert_sampler:sampleBatch()
   gstate = {focus='experts'}
end



local function switchnodeTest()
   local num_experts = 10
   local target_range = {0.1, 0.9}
   local experts = {}
   for i = 1,num_experts do
      table.insert(experts, dp.Neural{input_size=10,
                                      output_size=3,
                                      transfer=nn.Tanh()}
      )
   end
   local gater = dp.Sequential()
   gater:add(dp.Neural{input_size=10,
                       output_size=20,
                       transfer=nn.Tanh()}
   )
   gater:add(dp.Equanimous{input_size=20,
                           output_size=num_experts,
                           transfer=nn.Sigmoid()}
   )
   local sn = dp.SwitchNode{gater=gater, experts=experts}
end
