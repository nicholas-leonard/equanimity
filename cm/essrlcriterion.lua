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
   local args, n_sample, n_leaf, n_reinforce, n_eval, n_classes, 
         criterion, accumulator = xlua.unpack(
      {config},
      'ESSRLCriterion', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example during training.'},
      {arg='n_leaf', type='number',
       help='number of  leaf experts in the neural decision tree'},
      {arg='n_reinforce', type='number', 
       help='number of experts reinforced per example during training.'},
      {arg='n_eval', type='number', 
       help='number of experts chosen per example during evaluation.'},
      {arg='n_classes', type='number',
       help='number of classes'},
      {arg='criterion', type='nn.Criterion', default=nn.ClassNLLCriterion,
       help='Criterion to be used for optimizing winning experts'},
      {arg='accumulator', type='string', default='softmax'}
   )
   self._criterion = criterion
   self._n_reinforce = n_reinforce
   self._n_sample = n_sample
   self._n_leaf = n_leaf
   self._n_eval = n_eval
   self._n_classes = n_classes
   self._accumulator = accumulator
   -- statistics :
   --- monopoly : 
   ---- distribution of examples to experts
   self._reinforce_dist = torch.DoubleTensor(self._n_leaf)
   self._sample_dist = torch.DoubleTensor(self._n_leaf)
   ---- exponential moving average
   self._reinforce_ema = torch.DoubleTensor(self._n_leaf)
   self._sample_ema = torch.DoubleTensor(self._n_leaf)
   self._present_factor = 0.5 -- higher val discounts past vals faster
   ---- mean, standard deviation, min, max
   self._reinforce_stats = {}
   self._sample_stats = {}
   self:resetStatistics()
end

function ESSRLCriterion:resetStatistics()
   self._reinforce_dist:zero()
   self._sample_dist:zero()
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
   local input_origins = torch.LongTensor(unpack(size))
   local batch = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      if type(expert_idx) == 'number' then
         -- gather statistics
         self._sample_dist[expert_idx] 
            = self._sample_dist[expert_idx] + expert_ostate.act:size(1)
         self._sample_ema[expert_idx]
            = ((1-self._present_factor) * self._sample_ema[expert_idx])
            + (self._present_factor * expert_ostate.act:size(1))
         expert_ostate.act_double = expert_ostate.act:double()
         for i = 1, expert_ostate.batch_indices:size(1) do
            local batch_idx = expert_ostate.batch_indices[i]
            local example = batch[batch_idx] or {
               experts={}, acts={}, errors={},
               criteria={}, alphas={}, origins={}
            }        
            local act = expert_ostate.act_double[{i,{}}]
            local criterion = self._criterion()
            local err = criterion:forward(act, targets[batch_idx])
            table.insert(example.experts, expert_idx)
            table.insert(example.acts, act)
            table.insert(example.criteria, criterion)
            table.insert(example.errors, err)
            table.insert(example.alphas, expert_ostate.alphas[i])
            table.insert(example.origins, i)
            batch[batch_idx] = example
         end
      end
   end
   for batch_idx, example in pairs(batch) do
      -- sort each example's experts by ascending error
      local example_errors, idxs = torch.DoubleTensor(
         example.errors
      ):sort()
      assert(example_errors:size(1) == self._n_sample)
      local example_acts = torch.DoubleTensor(
         #example.acts, self._n_classes
      )
      for sample_idx, sample_acts in pairs(example.acts) do
         example_acts[{sample_idx, {}}] = sample_acts
      end
      local example_experts = torch.LongTensor(example.experts)
      local example_alphas = torch.DoubleTensor(example.alphas)
      local example_origins = torch.LongTensor(example.origins)
      input_acts[{batch_idx, {}, {}}] = example_acts:index(1, idxs)
      input_experts[{batch_idx,{}}] = example_experts:index(1, idxs)
      input_alphas[{batch_idx,{}}] = example_alphas:index(1, idxs)
      input_origins[{batch_idx,{}}] = example_origins:index(1, idxs)
      -- reorder criteria and keep only winning for backward pass
      local criteria = {}
      for sample_idx = 1,input_experts:size(2) do
         if sample_idx > self._n_reinforce then
            break
         end
         table.insert(criteria, example.criteria[idxs[sample_idx]])
      end
   end
   -- For each example, reinforce winning experts (those that have 
   -- least error) by allowing them to learn and backward-propagating
   -- the indices of winning experts for use by the gaters which will
   -- reinforce the winners.
   -- keep n_reinforce winners
   local win_input_acts = input_acts[{{},{1,self._n_reinforce},{}}]
   local win_input_experts = input_experts[{{},{1,self._n_reinforce}}]
   local win_input_alphas = input_alphas[{{},{1,self._n_reinforce}}]
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
   local criterion = self._criterion()
   local output_error = criterion:forward(outputs, targets)
   -- store for backward pass
   self._input_acts = input_acts
   self._input_experts = input_experts
   self._input_origins = input_origins
   self._win_input_experts = win_input_experts
   self._batch = batch
   return output_error, outputs
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
         --print('act', expert_ostate.act_double)
         --print('alphas', expert_ostate.alphas)
         for i = 1, expert_ostate.batch_indices:size(1) do
            local batch_idx = expert_ostate.batch_indices[i]
            local example = batch[batch_idx] or {acts={}, alphas={}}  
            table.insert(example.acts, expert_ostate.act_double[{i,{}}])
            table.insert(example.alphas, expert_ostate.alphas[i])
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
   --print('alphas', alphas)
   --print('outputs', outputs)
   -- measure error
   local criterion = self._criterion()
   local output_error = criterion:forward(outputs, targets)
   return output_error, outputs
end

function ESSRLCriterion:backward(expert_ostates, targets, indices)
   --regroup by experts
   local experts = {}
   for batch_idx = 1,self._input_experts:size(1) do
      local example = self._batch[batch_idx]
      for sample_idx = 1,self._input_experts:size(2) do
         local expert_idx = self._input_experts[{batch_idx,sample_idx}]
         local origin_idx = self._input_origins[{batch_idx,sample_idx}]
         local expert = experts[expert_idx] or {
            reinforce={}, grads={}, origins={}
         }
         table.insert(expert.origins, origin_idx)
         -- non-reinforce expert-example pairs have no grad (dont learn)
         local grad = false
         if sample_idx <= self._n_reinforce then
            local criterion = example.criteria[sample_idx]
            local act = self._input_acts:select(1,batch_idx):select(
               1, sample_idx
            )
            -- backprop through criterion
            grad = criterion:backward(act, targets[batch_idx])
            table.insert(expert.reinforce, origin_idx)
         end
         table.insert(expert.grads, grad)
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
      self._reinforce_ema[expert_idx] 
         = ((1-self._present_factor) * self._reinforce_ema[expert_idx])
         + (self._present_factor * #expert.reinforce)
      cstates[expert_idx] = {
         batch_indices = expert_ostate.batch_indices,
         reinforce_indices = expert.reinforce
      }
      local grad = expert_ostate.act_double:clone():zero()
      for batch_idx, example_grad in ipairs(expert.grads) do
         if example_grad then
            grad[{{batch_idx},{}}] = example_grad 
         end
      end
      --local grad = grad:type(expert_ostate.acts:type())
      expert_ostate.grad = grad:clone()
      expert_ostate.grad:indexCopy(1, idxs, grad)
      expert_ostate.grad = expert_ostate.grad:type(
         expert_ostate.act:type()
      )
   end
   return expert_ostates, cstates
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
      reinforce_ema = dp.distReport(self._reinforce_ema),
      sample_ema = dp.distReport(self._sample_ema)
   }
   local r = table.merge({}, report)
   r.reinforce.dist = table.tostring(r.reinforce.dist:storage():totable())
   r.sample.dist = table.tostring(r.sample.dist:storage():totable())
   r.reinforce_ema.dist = table.tostring(r.reinforce_ema.dist:storage():totable())
   r.sample_ema.dist = table.tostring(r.sample_ema.dist:storage():totable())
   print(r)
   return report
end
