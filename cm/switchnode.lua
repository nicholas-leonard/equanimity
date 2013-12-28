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
   local args, gater, experts = xlua.unpack(
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
   config.id = self:id():create('gater')
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

function SwitchNode:nExperts()
   return #self._experts
end

function SwitchNode:get(index)
   return self._experts[index]
end

function SwitchNode:_forward(cstate)
   --print(self:id():toString(), cstate)
   self.istate.act_double = self.istate.act:double()
   -- forward gater to get routes
   local gater_ostate = self._gater:forward{
      -- shallow copy to allow gater to compute its own grads
      input=table.copy(self.istate), global=self.gstate, carry=cstate
   }
   -- routes is a matrix of indices
   local routes = gater_ostate.routes
   -- alphas is a matrix of rows that some to one and weight
   local alphas = gater_ostate.alphas:clone()
   -- multiply these alphas by any previous alphas
   if self.istate.alphas then
      alphas:cmul(self.istate.alphas:resizeAs(alphas))
   end
   local input_act = self.istate.act_double
   self.istate.batch_indices = cstate.batch_indices
   local experts = {}
   -- TODO modify multinomial to return what the following loop returns
   -- accumulate example indices and alphas for each experts
   for example_idx = 1,routes:size(1) do
      for sample_idx = 1,routes:size(2) do
         local expert_idx = routes[{example_idx,sample_idx}]
         local expert = experts[expert_idx] or {examples={}, alphas={}}
         table.insert(expert.examples, example_idx)
         table.insert(expert.alphas, alphas[{example_idx,expert_idx}])
         experts[expert_idx] = expert
      end
   end   
   --self.ostate = {gater=gater_ostate}
   self.ostate = {}
   local output_cstates = {}
   for expert_idx, expert_branch in ipairs(self._experts) do
      local expert = experts[expert_idx]
      if expert then
         local expert_indices = torch.LongTensor(expert.examples) 
         local expert_cstate = {
            expert_indices=expert_indices,
            batch_indices=cstate.batch_indices:index(1, expert_indices)
         }
         -- forward batch and update i/o states
         local expert_ostate, expert_cstate = expert_branch:forward{
            input=input_act:index(1, expert_indices):type(
               self.istate.act:type()
            ), 
            global=self.gstate, carry=expert_cstate
         }
         -- save the alphas
         expert_ostate.alphas = torch.DoubleTensor(expert.alphas)
         table.merge(expert_ostate, expert_cstate)
         output_cstates[expert_idx] = expert_cstate
         self.ostate[expert_idx] = expert_ostate
      end
   end
   return output_cstates
end

function SwitchNode:_backward(cstates)
   local istate = {}
   local g_reinforce = {} -- indices for this layer's gater
   local i_reinforce = {} -- indices for previous layer
   self.istate.act_double = self.istate.act_double 
      or self.istate.act:double()
   local input_grad = self.istate.act:clone():zero()
   local expert_grad = self.istate.act_double:clone()
   for expert_idx, expert_cstate in pairs(cstates) do
      local expert_branch = self._experts[expert_idx]
      -- the indices of examples in the original node input batch
      local expert_indices = expert_branch.ostate.expert_indices
      local expert_ostate = self.ostate[expert_idx]
      -- indices of the input to the switch_node to reinforce
      if not _.isEmpty(expert_cstate.reinforce_indices) then
         local reinforce_indices = expert_indices:index(
            1, torch.LongTensor(expert_cstate.reinforce_indices)
         )
         g_reinforce[expert_idx] = reinforce_indices
         i_reinforce[expert_idx] = reinforce_indices:storage():totable()
      end
      local expert_istate = expert_branch:backward{
         output=expert_ostate, global=self.global,
         carry=cstates[expert_idx]
      }
      expert_grad:zero()
      -- TODO :
      -- compare speed of this to alternative (copy into expert igrad)
      expert_grad:indexCopy(
         1, expert_indices, expert_istate.grad:double()
      )
      -- accumulate input gradients from experts by addition
      input_grad:add(expert_grad:type(input_grad:type()))
   end 
   --[[ gater ]]--
   -- backward gater to get routes
   self._gater.ostate.reinforce_indices = g_reinforce
   local gater_istate = self._gater:backward{global=self.global}
   input_grad:add(gater_istate.grad)
   self.istate.grad = input_grad
   -- prepare reinforce indices for the next layer
   local cstate = {
      batch_indices = self.istate.batch_indices,
      reinforce_indices = _.uniq(_.concat(unpack(i_reinforce)))
   }
   return cstate
end

function SwitchNode:_evaluate(cstate)
   local routes = self._gater:evaluate(cstate)
   
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
end

function SwitchNode:__tostring__()
   return 'dp.SwitchNode'
end

function SwitchNode:report()
   -- merge reports
   local report = {
      experts={},
      gater=self._gater:report()
   }
   return report
end
