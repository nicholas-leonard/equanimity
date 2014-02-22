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
   local args, gater, experts, gater_grad_scale, zero_targets
      = xlua.unpack(
      {config},
      'SwitchNode', nil,
      {arg='gater', type='dp.Model'},
      {arg='experts', type='table'},
      {arg='gater_grad_scale', type='number', default=1,
       help='scales the gradient before it is added to gradients from'..
       ' experts which is fedback to previous layer'},
      {arg='zero_targets', type='boolean'}
   )
   config.typename = 'switchnode'
   parent.__init(self, config)
   self._gater = gater
   self._experts = experts
   self._models = _.concat({self._gater}, self._experts)
   self._gater_grad_scale = gater_grad_scale
   self._zero_targets = zero_targets
   -- alloc tensors
   self._sampled_targets = torch.DoubleTensor()
   self._gater_targets = torch.DoubleTensor()
   self._expert_grad = torch.DoubleTensor()
   self._class_targets = torch.DoubleTensor()
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
   config.id = self:id():create('gater')
   self._gater:setup(config)
   self._data_view = self._experts[1]:dataView()
end

function SwitchNode:nExperts()
   return #self._experts
end

function SwitchNode:get(index)
   return self._experts[index]
end

function SwitchNode:_forward(cstate)
   self.istate.act_double = self.istate.act:double()
   -- forward gater to get routes
   local gater_istate = {
      -- shallow copy to allow gater to compute its own grads
      input=table.copy(self.istate), global=self.gstate, carry=cstate
   }
   local gater_ostate 
   if self.gstate.evaluate then 
      gater_ostate = self._gater:evaluate(gater_istate)
   else 
      gater_ostate = self._gater:forward(gater_istate)
   end
   
   -- alphas is a matrix of rows that sum to one and weigh experts
   local alphas = gater_ostate.alphas:clone()
   -- P(E_l|X) = P(E_l|E_l-1,X)P(E_l-1|X) or P(A)= P(A|B)P(B)
   if self.istate.alphas then
      alphas:cmul(
         self.istate.alphas:reshape(
            self.istate.alphas:size(1), 1
         ):expandAs(alphas)
      )
   end
   local input_act = self.istate.act_double
   self.istate.batch_indices = cstate.batch_indices
   local experts = {}
   -- routes is a matrix of indices
   local routes = gater_ostate.routes
   -- TODO modify multinomial to return what the following loop returns
   -- accumulate example indices and alphas for each experts
   for example_idx = 1,routes:size(1) do
      for sample_idx = 1,routes:size(2) do
         local expert_idx = routes[{example_idx,sample_idx}]
         local expert = experts[expert_idx] 
            or {examples={}, alphas={}}
         table.insert(expert.examples, example_idx)
         table.insert(expert.alphas, alphas[{example_idx,expert_idx}])
         experts[expert_idx] = expert
      end
   end   
   
   self.ostate = {}
   local output_cstates = {}
   for expert_idx, expert in pairs(experts) do
      local expert_branch = self._experts[expert_idx]
      local expert_indices = expert.expert_indices or torch.LongTensor(expert.examples) 
      local expert_cstate = {
         expert_indices=expert_indices,
         batch_indices=cstate.batch_indices:index(1, expert_indices)
      }
      -- forward batch and update i/o states
      local expert_istate = {
         input=input_act:index(1, expert_indices):type(
            self.istate.act:type()
         ), 
         global=self.gstate, carry=expert_cstate
      }
      local expert_ostate, expert_cstate
      if self.gstate.evaluate then 
         expert_ostate, expert_cstate = expert_branch:evaluate(
            expert_istate
         )
      else
         expert_ostate, expert_cstate = expert_branch:forward(
            expert_istate
         )
      end
      -- save the alphas
      expert_ostate.alphas = torch.DoubleTensor(expert.alphas)
      table.merge(expert_ostate, expert_cstate)
      output_cstates[expert_idx] = expert_cstate
      self.ostate[expert_idx] = expert_ostate
   end
   return output_cstates
end

function SwitchNode:_backward(cstates)
   local istate = {}
   self._sampled_targets:resize(self._gater.ostate.act_double:size()):zero()
   self._class_targets:resize(self._gater.ostate.act_double:size(1)):zero()
   self.istate.act_double = self.istate.act_double or self.istate.act:double()
   local n_example = self.istate.act_double:size(1)
   if not self._input_grad then
      self._input_grad = self.istate.act:clone()
   end
   self._input_grad:resizeAs(self.istate.act):zero()
   self._expert_grad:resizeAs(self.istate.act_double)
   for expert_idx, expert_cstate in pairs(cstates) do
      local expert_branch = self._experts[expert_idx]
      -- the indices of examples in the original node input batch
      local expert_indices = expert_branch.ostate.expert_indices
      local expert_ostate = self.ostate[expert_idx]
      -- size-average param grads
      expert_cstate.scale = 1/expert_ostate.act:size(1)
      self._sampled_targets:select(2,expert_idx):indexCopy(
         1, expert_indices, expert_cstate.gater_targets
      )
      self._class_targets:indexCopy(
         1, expert_indices, expert_cstate.class_targets
      )
      local expert_istate = expert_branch:backward{
         output=expert_ostate, global=self.global, carry=expert_cstate
      }
      self._expert_grad:zero()
      self._expert_grad:indexCopy(1, expert_indices, expert_istate.grad:double())
      -- accumulate input gradients from experts by addition
      self._input_grad:add(self._expert_grad:type(self._input_grad:type()))
   end 
   --[[ gater ]]--
   -- before renormalize, get P(E_l-1|X) for parent gater targets
   local parent_gater_targets = self._sampled_targets:sum(2)
   -- p(E_l|E_l-1,X) = p(E_l|X)/P(E_l-1|X) or P(A|B)=P(A)/P(B)
   self._sampled_targets:cdiv(self._sampled_targets:sum(2):add(0.000000001):expandAs(self._sampled_targets))
   -- keep within target upper and lower bounds
   --self._sampled_targets:mul(0.8):add(0.1)
   self._gater_targets:resizeAs(self._gater.ostate.act_double)
   -- non-sampled expert-examples will have zero error (target = act)
   if self._zero_targets then
      self._gater_targets:fill(0)
   else
      self._gater_targets:copy(self._gater.ostate.act_double) 
   end
   self._gater.ostate.class_targets = self._class_targets
   for expert_idx, expert_cstate in pairs(cstates) do
      -- insert sampled (non-zero) p(E_l|E_l-1,X) targets into respective slots
      local expert_indices = self._experts[expert_idx].ostate.expert_indices
      self._gater_targets:select(2, expert_idx):indexCopy(
         1, expert_indices, self._sampled_targets:select(2, expert_idx):index(1, expert_indices)
      )
   end
   self._gater.ostate.gater_targets = self._gater_targets
   local gater_istate = self._gater:backward{
      global=self.gstate, carry={scale=1/n_example}
   }
   self._input_grad:add(self._gater_grad_scale, gater_istate.grad)
   self.istate.grad = self._input_grad
   local cstate = {
      batch_indices = self.istate.batch_indices,
      gater_targets = parent_gater_targets:reshape(parent_gater_targets:size(1)),
      class_targets = self._class_targets
   }
   return cstate
end

function SwitchNode:zeroGradParameters()
  for i=1,#self._models do
     self._models[i]:zeroGradParameters()
  end
end

function SwitchNode:reset(stdv)
   for i=1,#self._models do
      self._models[i]:reset(stdv)
   end
end

function SwitchNode:parameters()
   error"NotImplementedError"
end

function SwitchNode:__tostring__()
   return 'dp.SwitchNode'
end
