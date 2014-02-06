------------------------------------------------------------------------
--[[ UGCriterion ]]--
-- Unsupervised Gater (AutoEncoder + Kmeans)
-- takes a table of tables as input.
-- Gater is the specializing force. 
------------------------------------------------------------------------
local UGCriterion, parent = torch.class("nn.UGCriterion", "nn.ESSRLCriterion")

function UGCriterion:__init(config)
   config = config or {}
   local args, accumulator = xlua.unpack(
      {config},
      'UGCriterion', 'Unsupervised Gater (AutoEncoder + Kmeans)',
      {arg='accumulator', type='string', default='normalize'}
   )
   config.accumulator = accumulaotr
   parent.__init(self, config)
end

function UGCriterion:forward(expert_ostates, targets, indices)
   assert(type(expert_ostates) == 'table')
   assert(torch.isTensor(targets))
   assert(torch.isTensor(indices))
   local n_example = indices:size(1)
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
         table.insert(example.errors, self._criterion:forward(act, target))   
         batch[batch_idx] = example      
      end
   end
   -- evaluate
   local output_error, outputs = self:_evaluate(batch, targets)
   -- gradients grouped by expert
   local expert_cstates = {}
   for expert_idx, expert_ostate in pairs(expert_ostates) do
      -- backprop through criterion
      expert_ostate.grad = self._criterion:backward(
         expert_ostate.act_double, targets:index(1, expert_ostate.batch_indices)
      ):clone()
      -- grad * P(E_l|X)
      local p = expert_ostate.alphas
      assert(p:dim() == 1)
      -- even if a cluster monopolises, it cannot have a total grad  
      -- weight greater than size of its input batch
      expert_ostate.grad:cmul(p:resize(p:size(1),1):expandAs(expert_ostate.grad))
      expert_cstates[expert_idx] = {
         batch_indices = expert_ostate.batch_indices,
      }
       expert_ostate.grad = expert_ostate.grad:type(expert_ostate.act:type())
   end
   return output_error, outputs, expert_ostates, expert_cstates
end

