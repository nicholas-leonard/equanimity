------------------------------------------------------------------------
--[[ ESSRLCriterion ]]--
-- Equanimous Sparse Supervised Reinforcement Learning
-- takes a table of tables as input.
-- TODO : 
-- build as container or model?
-- statistics : Reinit after epoch, gen report
--- distribution of error in experts
------------------------------------------------------------------------
local ESSRLCriterion, parent = torch.class("nn.ESSRLCriterion")

function ESSRLCriterion:__init(config)
   config = config or {}
   local args, n_sample, n_leaf, n_reinforce, n_backprop, n_eval, 
         n_classes, criterion, accumulator, backprop_pad
      = xlua.unpack(
      {config},
      'ESSRLCriterion', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example during training.'},
      {arg='n_leaf', type='number',
       help='number of  leaf experts in the neural decision tree'},
      {arg='n_reinforce', type='number', 
       help='number of experts reinforced per example during training.'},
      {arg='n_backprop', type='number', 
       help='number of experts backproped per example during training.'},
      {arg='n_eval', type='number', 
       help='number of experts chosen per example during evaluation.'},
      {arg='n_classes', type='number',
       help='number of classes'},
      {arg='criterion', type='nn.Criterion', default=nn.ClassNLLCriterion,
       help='Criterion to be used for optimizing winning experts'},
      {arg='accumulator', type='string', default='softmax'},
      {arg='backprop_pad', type='number', default=0,
       help='number of experts with least error to ignore before '..
       'backpropagating through n_backprop experts. Used to keep '..
       'lesser experts in the game.'}
   )
   -- we expect the criterion to be stateless (we use it as a function)
   self._criterion = criterion()
   -- stop torch from scaling grads based on batch_size (we do so later)
   self._criterion.sizeAverage = false
   self._n_reinforce = n_reinforce
   self._n_backprop = n_backprop or n_reinforce
   self._n_sample = n_sample
   self._n_leaf = n_leaf
   self._n_eval = n_eval
   self._n_classes = n_classes
   self._accumulator = accumulator
   self._backprop_pad = backprop_pad
   -- statistics :
   --- monopoly : 
   ---- distribution of examples to experts
   self._reinforce_dist = torch.DoubleTensor(self._n_leaf)
   self._sample_dist = torch.DoubleTensor(self._n_leaf)
   --- specialization :
   ---- records distribution of backprops (train) or samples (eval)
   self._spec_matrix = torch.DoubleTensor(self._n_leaf, self._n_classes) 
   self._err_matrix = torch.DoubleTensor(self._n_leaf, self._n_classes) 
   self:resetStatistics()
end

function ESSRLCriterion:resetStatistics()
   self._reinforce_dist:zero()
   self._sample_dist:zero()
   self._spec_matrix:zero()
   self._err_matrix:zero()
end

function ESSRLCriterion:forward(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local n_example = indices:size(1)
   local act_size = {n_example, self._n_sample, self._n_classes}
   local input_acts = torch.DoubleTensor(unpack(act_size))
   local size = {n_example, self._n_sample}
   local input_experts = torch.LongTensor(unpack(size))
   local input_alphas = torch.DoubleTensor(unpack(size))
   -- original indices of example in expert mini-batch
   local input_origins = torch.LongTensor(unpack(size))
   local batch = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      if type(expert_idx) == 'number' then
         -- gather statistics
         self._sample_dist[expert_idx] 
            = self._sample_dist[expert_idx] + expert_ostate.act:size(1)
         expert_ostate.act_double = expert_ostate.act:double()
         for i = 1, expert_ostate.batch_indices:size(1) do
            local batch_idx = expert_ostate.batch_indices[i]
            local example = batch[batch_idx] or {
               experts={}, acts={}, errors={}, targets={},
               criteria={}, alphas={}, origins={}
            }        
            local act = expert_ostate.act_double[{i,{}}]
            local target = targets[batch_idx] 
            local err = self._criterion:forward(act, target)
            table.insert(example.experts, expert_idx)
            table.insert(example.targets, target)
            table.insert(example.acts, act)
            table.insert(example.errors, err)
            table.insert(example.alphas, expert_ostate.alphas[i])
            table.insert(example.origins, i)
            batch[batch_idx] = example
         end
      end
   end
   for batch_idx, example in pairs(batch) do
      -- sort each example's experts by descending err
      local example_errors, idxs = torch.DoubleTensor(
         example.errors
      ):sort(1, false)
      assert(example_errors:size(1) == self._n_sample)
      local example_acts = torch.DoubleTensor(
         #example.acts, self._n_classes
      )
      for sample_idx, sample_acts in pairs(example.acts) do
         example_acts[{sample_idx, {}}] = sample_acts
      end
      local example_targets 
         = torch.LongTensor(example.targets):index(1, idxs)
      local target = example_targets[1+self._backprop_pad]
      local example_experts 
         = torch.LongTensor(example.experts):index(1, idxs)
      local expert_idx = example_experts[1+self._backprop_pad]
      -- gater stats
      self._spec_matrix[{expert_idx, target}] 
            = self._spec_matrix[{expert_idx, target}] + 1 
      local err = example_errors[1+self._backprop_pad]
      self._err_matrix[{expert_idx, target}] 
            = self._err_matrix[{expert_idx, target}] + err
      local example_alphas = torch.DoubleTensor(example.alphas)
      local example_origins = torch.LongTensor(example.origins)
      input_acts[{batch_idx, {}, {}}] = example_acts:index(1, idxs)
      input_experts[{batch_idx,{}}] = example_experts
      input_alphas[{batch_idx,{}}] = example_alphas:index(1, idxs)
      input_origins[{batch_idx,{}}] = example_origins:index(1, idxs)
   end
   -- For each example, reinforce winning experts (those that have 
   -- least error) by allowing them to learn and backward-propagating
   -- the indices of winning experts for use by the gaters which will
   -- reinforce the winners.
   -- keep n_reinforce winners
   local sub = {1,self._n_eval}
   local win_input_acts = input_acts[{{},sub,{}}]
   local win_input_alphas = input_alphas[{{},sub}]
   local alphas
   if self._accumulator == 'softmax' then
      -- normalize alphas using softmax
      alphas = torch.exp(win_input_alphas)
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   elseif self._accumulator == 'normalize' then 
      -- normalize alphas to sum to one
      local sum = win_input_alphas:sum(2):expandAs(win_input_alphas)
      alphas = torch.cdiv(win_input_alphas,sum)
   elseif self._accumulator == 'truncate' then 
      -- normalize alphas to sum to one but truncate min val
      alphas = win_input_alphas:add(
         -win_input_alphas:min(2):expandAs(win_input_alphas)
      )
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   else
      error"Unknown accumulator"
   end
   -- alpha-weighted mean of activations to get outputs
   local size = alphas:size():totable()
   size[3] = 1
   local outputs = torch.cmul(
      torch.reshape(alphas, unpack(size)):expandAs(win_input_acts), 
      win_input_acts
   ):sum(2)[{{},1,{}}]
   -- measure error
   local output_error = self._criterion:forward(outputs, targets)
   -- store for backward pass
   self._input_acts = input_acts
   self._input_experts = input_experts
   self._input_origins = input_origins
   self._batch = batch
   return output_error, outputs
end

function ESSRLCriterion:backward(expert_ostates, targets, indices)
   --regroup by experts
   local experts = {}
   for batch_idx = 1,self._input_experts:size(1) do
      local example = self._batch[batch_idx]
      local n_sample = self._input_experts:size(2)
      for sample_idx = 1,n_sample do
         local expert_idx = self._input_experts[{batch_idx,sample_idx}]
         local origin_idx = self._input_origins[{batch_idx,sample_idx}]
         local expert = experts[expert_idx] or {
            reinforce={}, grads={}, origins={}, backprop={}
         }
         table.insert(expert.origins, origin_idx)
         -- focus backprop on first experts (see forward() for order)
         -- some expert-example pairs have no grad (dont learn)
         if sample_idx > self._backprop_pad 
               and sample_idx <= self._n_backprop then
            local act = self._input_acts:select(1, batch_idx):select(
               1, sample_idx
            )
            -- backprop through criterion
            expert.grads[origin_idx] = self._criterion:backward(
               act, targets[batch_idx]
            ):clone()
            table.insert(expert.backprop, origin_idx)
         end
         -- reinforce stronger experts in gater
         if sample_idx <= self._n_reinforce then
            table.insert(expert.reinforce, origin_idx)
         end
         experts[expert_idx] = expert
      end
   end
   local cstates = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      local expert = experts[expert_idx]
      -- reorder to original order of examples in batch
      local idxs = torch.LongTensor(expert.origins)
      -- gather statistics
      self._reinforce_dist[expert_idx] 
         = self._reinforce_dist[expert_idx] + #expert.reinforce
      -- gradients
      local grad = expert_ostate.act_double:clone():zero()
      for origin_idx, example_grad in pairs(expert.grads) do
         grad[{{origin_idx},{}}] = example_grad 
      end
      cstates[expert_idx] = {
         batch_indices = expert_ostate.batch_indices,
         reinforce_indices = expert.reinforce,
         backprop_indices = expert.backprop
      }
      expert_ostate.grad = grad:type(expert_ostate.act:type())
   end
   return expert_ostates, cstates
end

function ESSRLCriterion:expertFocus(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   -- per batch, we should reinforce about as many expert-example 
   -- pairs as during the exampleFocus phase, 
   -- i.e. n_reinforce * batch_size out of n_sample * batch_size
   local reinforce_factor = self._n_reinforce/self._n_sample
   local backprop_factor = self._n_backprop/self._n_sample
   local cstates = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      local batch_indices = expert_ostate.batch_indices
      local n_example = batch_indices:size(1)
      local expert_acts = expert_ostate.act:double()
      -- measure individual errors 
      local expert_targets = targets:index(1, batch_indices)
      local expert_errors = expert_targets:clone()
      for sample_idx = 1,n_example do
         expert_errors[sample_idx] = self._criterion:forward(
            expert_acts[{sample_idx,{}}], expert_targets[sample_idx]
         )
      end
      -- sort each expert's examples by ascending error
      local expert_errors, idxs = expert_errors:sort(1)
      -- have gater learn examples with least error
      local n_reinforce = math.max(
         1, math.floor(reinforce_factor * n_example)
      )
      -- have expert learn examples with least error (specialize)
      local n_backprop = math.max(
         1, math.floor(backprop_factor * n_example)
      )
      local reinforce_indices = idxs[{{1,n_reinforce}}]
      local backprop_indices = idxs[{{1,n_backprop}}]
      cstates[expert_idx] = {
         batch_indices = expert_ostate.batch_indices,
         reinforce_indices = reinforce_indices:storage():totable(),
         backprop_indices = backprop_indices:storage():totable()
      }
      
      -- backpropagate criteria of least-error examples for expert
      local expert_grads = expert_acts:clone():zero()
      local backprop_grads = self._criterion:backward(
         expert_acts:index(1, backprop_indices), 
         expert_targets:index(1, backprop_indices)
      ):clone()
      expert_grads:indexCopy(1, backprop_indices, backprop_grads)
      expert_ostate.grad = expert_grads:type(expert_ostate.act:type())
   end
   return expert_ostates, cstates
end

function ESSRLCriterion:evaluate(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local n_example = indices:size(1)
   local act_size = {n_example, self._n_sample, self._n_classes}
   local input_acts = torch.DoubleTensor(unpack(act_size))
   local input_alphas = torch.DoubleTensor(n_example, self._n_sample)
   local batch = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      if type(expert_idx) == 'number' then
         -- gather statistics
         self._sample_dist[expert_idx] 
            = self._sample_dist[expert_idx] + expert_ostate.act:size(1)
         expert_ostate.act_double = expert_ostate.act:double()
         for i = 1, expert_ostate.batch_indices:size(1) do
            local batch_idx = expert_ostate.batch_indices[i]
            local example = batch[batch_idx] 
               or {acts={},alphas={},errors={},targets={},experts={}}  
            local act = expert_ostate.act_double[{i,{}}]
            table.insert(example.acts, act)
            table.insert(example.alphas, expert_ostate.alphas[i])
            -- for stats gathering
            local target = targets[batch_idx]
            table.insert(
               example.errors, self._criterion:forward(act, target)
            )
            table.insert(example.targets, target)
            table.insert(example.experts, expert_idx)
            batch[batch_idx] = example
         end
      end
   end
   for batch_idx, example in pairs(batch) do
      -- sort each example's experts by descending alpha
      local example_alphas, idxs = torch.DoubleTensor(
         example.alphas
      ):sort(1,true)
      assert(example_alphas:size(1) == self._n_sample)
      local example_acts = torch.DoubleTensor(
         #example.acts, self._n_classes
      )
      for sample_idx, sample_acts in pairs(example.acts) do
         example_acts[{sample_idx, {}}] = sample_acts
      end
      input_acts[{batch_idx, {}, {}}] = example_acts:index(1, idxs)
      input_alphas[{batch_idx,{}}] = example_alphas
      -- gather stats
      local target = example.targets[idxs[1]]
      local expert_idx = example.experts[idxs[1]]
      self._spec_matrix[{expert_idx, target}] 
            = self._spec_matrix[{expert_idx, target}] + 1 
      local err = example.errors[idxs[1]]
      self._err_matrix[{expert_idx, target}] 
            = self._err_matrix[{expert_idx, target}] + err
   end
   local win_input_acts = input_acts[{{},{1,self._n_eval},{}}]
   local win_input_alphas = input_alphas[{{},{1,self._n_eval}}]
   local alphas
   if self._accumulator == 'softmax' then
      -- normalize alphas using softmax
      alphas = torch.exp(win_input_alphas)
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   elseif self._accumulator == 'normalize' then 
      -- normalize alphas to sum to one
      local sum = win_input_alphas:sum(2):expandAs(win_input_alphas)
      alphas = torch.cdiv(win_input_alphas,sum)
   elseif self._accumulator == 'truncate' then 
      -- normalize alphas to sum to one but truncate min val
      alphas = win_input_alphas:add(
         -win_input_alphas:min(2):expandAs(win_input_alphas)
      )
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   else
      error"Unknown accumulator"
   end
   -- alpha-weighted mean of activations to get outputs
   local size = alphas:size():totable()
   size[3] = 1
   local outputs = torch.cmul(
      torch.reshape(alphas, unpack(size)):expandAs(win_input_acts), 
      win_input_acts
   ):sum(2)[{{},1,{}}]
   -- measure error
   local output_error = self._criterion:forward(outputs, targets)
   return output_error, outputs
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

function ESSRLCriterion:report()
   local report = {
      reinforce = dp.distReport(self._reinforce_dist),
      sample = dp.distReport(self._sample_dist),
      spec = self._spec_matrix,
      expert_error = self._err_matrix:sum(2):cdiv(self._spec_matrix:sum(2))
   }
   local r = table.merge({}, report)
   r.reinforce.dist = table.tostring(r.reinforce.dist:storage():totable())
   r.sample.dist = table.tostring(r.sample.dist:storage():totable())
   print(r)
   print('specialization matrix')
   print(self._spec_matrix)
   print('error matrix')
   self._spec_matrix:add(0.000001)
   print(self._err_matrix:sum(2):cdiv(self._spec_matrix:sum(2)))
   print(self._err_matrix:sum(1):cdiv(self._spec_matrix:sum(1)))
   return report
end
