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
   local args, n_sample, n_leaf, n_eval, n_classes, criterion, 
      accumulator, sparsity_factor, antispec = xlua.unpack(
      {config},
      'ESSRLCriterion', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example during training.'},
      {arg='n_leaf', type='number',
       help='number of  leaf experts in the neural decision tree'},
      {arg='n_eval', type='number', 
       help='number of experts chosen per example during evaluation.'},
      {arg='n_classes', type='number',
       help='number of classes'},
      {arg='criterion', type='nn.Criterion', default=nn.ClassNLLCriterion(),
       help='Criterion to be used for optimizing winning experts'},
      {arg='accumulator', type='string', default='softmax'},
      {arg='sparsity_factor', type='number', default=-1,
       help='increases the sparsity of the equanimous distributions'},
      {arg='antispec', type='boolean', default=false,
       help='backprop through worst examples in each expert'}
   )
   -- we expect the criterion to be stateless (we use it as a function)
   self._criterion = criterion
   -- stop torch from scaling grads based on batch_size (we do so later)
   self._criterion.sizeAverage = false
   self._n_sample = n_sample
   self._n_leaf = n_leaf
   self._n_eval = n_eval
   self._n_classes = n_classes
   self._accumulator = accumulator
   self._sparsity_factor = sparsity_factor
   self._antispec = antispec
   -- statistics :
   ---- records distribution of backprops (train) or samples (eval)
   self._spec_matrix = torch.DoubleTensor(self._n_leaf, self._n_classes) 
   self._err_matrix = torch.DoubleTensor(self._n_leaf, self._n_classes) 
   self:resetStatistics()
   -- allocate tensors
   self._input_acts = torch.DoubleTensor()
   self._input_alphas = torch.DoubleTensor()
   -- original indices of example in expert mini-batch
   self._output_errors = torch.DoubleTensor()
   self._backprop_weights = torch.DoubleTensor()
   self._gater_targets = torch.DoubleTensor()
   self._expert_errors = torch.DoubleTensor()
end

function ESSRLCriterion:resetStatistics()
   self._spec_matrix:zero()
   self._err_matrix:zero()
end

function ESSRLCriterion:forward(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local n_example = indices:size(1)
   self._output_errors:resize(n_example, self._n_sample):zero()
   self._backprop_weights:resize(n_example, self._n_leaf):zero()
   self._gater_targets:resize(n_example, self._n_leaf):zero()
   local batch = {}
   --[[ group by example ]]--
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      expert_ostate.act_double = expert_ostate.act:double()
      local expert_size = expert_ostate.batch_indices:size(1)
      self._expert_errors:resize(expert_size):zero()
      for i = 1, expert_size do
         local batch_idx = expert_ostate.batch_indices[i]
         local example = batch[batch_idx] or {
            acts={}, targets={}, alphas={}, experts={}, errors={}, grads={}
         }
         table.insert(example.alphas, expert_ostate.alphas[i])
         table.insert(example.experts, expert_idx)   
         local act = expert_ostate.act_double[{i,{}}]
         table.insert(example.acts, act)
         local target = targets[batch_idx]
         table.insert(example.targets, target)
         local err = self._criterion:forward(act, target)
         self._expert_errors[i] = err
         table.insert(example.errors, err)   
         batch[batch_idx] = example      
      end
      self._expert_errors:add(-self._expert_errors:min()) --make positive
      self._expert_errors:add(self._expert_errors:mean()*0.1) --epsilon cushion
      self._expert_errors:div(self._expert_errors:sum()+0.00001) -- normalize to sum to one
      self._backprop_weights:select(2, expert_idx):indexCopy(
         1, expert_ostate.batch_indices, dp.reverseDist(self._expert_errors)
      )
   end
   -- evaluate
   local output_error, outputs = self:_evaluate(batch, targets)
   -- make backprop grads equanimous
   for i = 1,3 do
      self._backprop_weights:cdiv(self._backprop_weights:sum(2):add(0.00001):expandAs(self._backprop_weights))
      if self._sparsity_factor > 0 then
         --increases the sparsity of the distribution
         self._backprop_weights:pow(self._sparsity_factor)
      end
      self._backprop_weights:cdiv(self._backprop_weights:sum(1):add(0.00001):expandAs(self._backprop_weights))
   end
   self._gater_targets:copy(self._backprop_weights)
   -- matrix probabilities of sampling leaf-expert given example
   self._gater_targets:cdiv(self._gater_targets:sum(2):expandAs(self._gater_targets))
   -- gradients grouped by expert
   local expert_cstates = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      -- backprop through criterion
      expert_ostate.grad = self._criterion:backward(
         expert_ostate.act_double, targets:index(1, expert_ostate.batch_indices)
      ):clone()
      -- weigh gradients by the equanimous distribution of reverse error
      local backprop_weights = self._backprop_weights:select(2, expert_idx):index(1, expert_ostate.batch_indices)
      if self._antispec then
         backprop_weights = dp.reverseDist(backprop_weights)
      end
      expert_ostate.grad:cmul(backprop_weights:reshape(backprop_weights:size(1), 1):expandAs(expert_ostate.grad))
      -- scale weights to full batch of examples
      expert_ostate.grad:mul(expert_ostate.batch_indices:size(1))
      -- target probability distributions for gater(s) 
      local gater_targets = self._gater_targets:select(2, expert_idx):index(1, expert_ostate.batch_indices)
      expert_cstates[expert_idx] = {
         batch_indices = expert_ostate.batch_indices,
         gater_targets = gater_targets
      }
      expert_ostate.grad = expert_ostate.grad:type(expert_ostate.act:type())
   end
   return output_error, outputs, expert_ostates, expert_cstates
end

function ESSRLCriterion:evaluate(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local n_example = indices:size(1)
   local batch = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      expert_ostate.act_double = expert_ostate.act:double()
      for i = 1, expert_ostate.batch_indices:size(1) do
         local batch_idx = expert_ostate.batch_indices[i]
         local example = batch[batch_idx] or {
            acts={},alphas={},errors={},targets={},experts={}
         }  
         local act = expert_ostate.act_double[{i,{}}]
         table.insert(example.acts, act)
         table.insert(example.alphas, expert_ostate.alphas[i])
         -- for stats gathering
         local target = targets[batch_idx]
         table.insert(example.errors, self._criterion:forward(act, target))
         table.insert(example.targets, target)
         table.insert(example.experts, expert_idx)
         batch[batch_idx] = example
      end
   end
   return self:_evaluate(batch, targets)
end

function ESSRLCriterion:_evaluate(batch, targets)
   self._input_acts:resize(#batch, self._n_sample, self._n_classes):zero()
   self._input_alphas:resize(#batch, self._n_sample):zero()
   for batch_idx, example in pairs(batch) do
      -- sort each example's experts by descending alpha
      local example_alphas, idxs = torch.DoubleTensor(example.alphas):sort(1,true)
      assert(example_alphas:size(1) == self._n_sample)
      local example_acts = torch.DoubleTensor(#example.acts, self._n_classes)
      for sample_idx, sample_acts in pairs(example.acts) do
         example_acts[{sample_idx, {}}] = sample_acts
      end
      self._input_acts[{batch_idx, {}, {}}] = example_acts:index(1, idxs)
      self._input_alphas[{batch_idx,{}}] = example_alphas
      -- gather stats
      local target = example.targets[idxs[1]]
      local expert_idx = example.experts[idxs[1]]
      self._spec_matrix[{expert_idx, target}] = self._spec_matrix[{expert_idx, target}] + 1 
      self._err_matrix[{expert_idx, target}] = self._err_matrix[{expert_idx, target}] + example.errors[idxs[1]]
   end
   local win_input_acts = self._input_acts[{{},{1,self._n_eval},{}}]
   local win_input_alphas = self._input_alphas[{{},{1,self._n_eval}}]
   local alphas
   if self._accumulator == 'softmax' then
      -- normalize alphas using softmax
      alphas = torch.exp(win_input_alphas)
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   elseif self._accumulator == 'normalize' then 
      -- normalize alphas to sum to one
      local sum = win_input_alphas:sum(2):expandAs(win_input_alphas)
      alphas = torch.cdiv(win_input_alphas,sum)
   else
      error"Unknown accumulator"
   end
   -- alpha-weighted mean of activations to get outputs
   local size = alphas:size():totable(); size[3] = 1;
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
      spec = self._spec_matrix,
      expert_error = self._err_matrix:cdiv(torch.add(self._spec_matrix,0.00001))
   }
   return report
end
