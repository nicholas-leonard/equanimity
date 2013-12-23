------------------------------------------------------------------------
--[[ ESSRLCriterion ]]--
-- Equanimous Sparse Supervised Reinforcement Learning
-- takes a table of tables as input.
-- TODO : build as container or model?
------------------------------------------------------------------------
local ESSRLCriterion, parent = torch.class("nn.ESSRLCriterion")

function ESSRLCriterion:__init(config)
   config = config or {}
   local args, n_sample, n_reinforce, n_eval, n_classes, criterion,
         accumulator
      = xlua.unpack(
      {config},
      'ESSRLCriterion', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example during training.'},
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
   self._n_eval = n_eval
   self._n_classes = n_classes
   self._accumulator = accumulator
   self._type = type
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
   local output_errors = torch.DoubleTensor(unpack(size))
   local batch = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      if type(expert_idx) == 'number' then
         for i = 1, expert_ostate.indices:size(1) do
            local batch_idx = expert_ostate.batch_indices[i]
            local example_idx = expert_ostate.indices[i]
            local example = batch[batch_idx] 
               or {experts={},acts={},errors={},criteria={},alphas={}}
            local act = expert.ostate.act[{i,{}}]
            local criterion = self._criterion()
            local err = criterion:forward(act, targets[batch_idx])
            table.insert(example.experts, expert_idx)
            table.insert(example.acts, act)
            table.insert(example.criteria, criterion)
            table.insert(example.errors, err)
            table.insert(example.alphas, expert_ostate.alphas[i])
            batch[batch_idx] = example
         end
      end
   end
   for batch_idx, example in pairs(batch) do
      -- sort each example's experts by ascending error
      local example_error, idxs = torch.DoubleTensor(example.errors):sort()
      local example_acts = input_acts[{batch_idx, {}, {}}]
      for sample_idx, acts in pairs(example.acts) do
         local sample_acts = torch.DoubleTensor(acts)
         example_acts[{sample_idx, {}}] = sample_acts:index(1, idxs)
      end
      local example_experts = torch.LongTensor(example.experts)
      local example_alphas = torch.DoubleTensor(example.alphas)
      input_experts[{batch_idx,{}}] = example_experts:index(1, idxs)
      input_alphas[{batch_idx,{}}] = example_alphas:index(1, idxs)
      output_errors[{batch_idx,{}}] = example_error:index(1, idxs)
   end
   -- For each example, reinforce winning experts (those that have 
   -- least error) by allowing them to learn and backward-propagating
   -- the indices of winning experts for use by the gaters which will
   -- reinforce the winners.
   -- keep n_reinforce winners
   local win_input_acts = input_acts[{{},{1,self._n_reinforce},{}}]
   local win_input_experts = input_experts[{{},{1,self._n_reinforce}}]
   local win_input_alphas = input_alphas[{{},{1,self._n_reinforce}}]
   local win_output_errors = output_errors[{{},{1,self._n_reinforce}}]
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
   return output_error, outputs
end


function ESSRLCriterion:backward(istates, targets, indices)
   for input_id, input_state in pairs(istates) do
      input_state.reinforce = {}
      input_state.grad = 0
   end
   -- compute gradients for the winning example-expert pairs
   for id, example in pairs(self._examples) do
      for i, winner in ipairs(example.winners) do
         table.insert(istates[winner.id].reinforce, winner.id)
      end
   end
end

function ESSRLCriterion:evaluate(istates, targets, indices)
   
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
