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
   -- forward gater to get routes
   local gater_ostate = self._gater:forward{
      -- shallow copy to allow gater to compute its own grads
      input=table.copy(self.istate), global=self.gstate, carry=cstate
   }
   -- routes is a matrix of indices
   local routes = gater_ostate.routes
   -- alphas is a matrix of rows that some to one and weight
   local alphas = gater_ostate.alphas
   -- multiply these alphas by any previous alphas
   if self.istate.alphas then
      alphas:cmul(self.istate.alphas)
   end
   local input_act = self.istate.act
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
   for expert_idx, expert_branch in ipairs(self._experts) do
      local expert = experts[expert_idx]
      if expert then
         -- create a tensor-batch examples for the expert
         local expert_indices = torch.LongTensor(expert.examples) 
         local expert_cstate = {
            expert_indices=expert_indices,
            batch_indices=cstate.batch_indices:index(1, expert_indices)
         }
         -- forward batch and update i/o states
         local expert_ostate, expert_cstate = expert_branch:forward{
            input=input_act:index(1, expert_indices), 
            global=self.gstate, 
            carry=expert_cstate
         }
         -- save the alphas
         expert_ostate.alphas = torch.DoubleTensor(expert.alphas)
         table.merge(expert_ostate, expert_cstate)
         self.ostate[expert_idx] = expert_ostate
      else
         -- is this really necessary?
         expert_branch.istate = {}
         expert_branch.ostate = {}
      end
   end
end

function SwitchNode:_backward(cstates)
   local istate = {}
   local reinforce_indices = {}
   local input_grad = self.istate.act:clone():zero()
   local expert_grad = input_grad:clone()
   for expert_idx, expert_cstate in pairs(cstates) do
      local expert_branch = self._experts[expert_idx]
      -- the indices of examples in the original node input batch
      local expert_indices = expert_branch.ostate.expert_indices
      local expert_ostate = self.ostate[expert_idx]
      -- indices of the input to the switch_node to reinforce
      if not _.isEmpty(expert_cstate.reinforce_indices) then
         reinforce_indices[expert_idx] = expert_indices:index(
            1, torch.LongTensor(expert_cstate.reinforce_indices)
         )
      end
      local expert_istate = expert_branch:backward{
         output=expert_ostate, global=self.global,
         carry=cstates[expert_idx]
      }
      expert_grad:zero()
      -- TODO :
      -- compare speed of this to alternative (copy into expert igrad)
      expert_grad:indexCopy(1, expert_indices, expert_istate.grad)
      -- accumulate input gradients from experts by addition
      input_grad:add(expert_grad)
   end 
   --[[ gater ]]--
   -- backward gater to get routes
   self._gater.ostate.reinforce_indices = reinforce_indices
   local gater_istate = self._gater:backward{global=self.global}
   input_grad:add(gater_istate.grad)
   self.istate.grad = input_grad
   -- prepare reinforce indices for the next layer
   local cstate = {
      batch_indices = self.istate.batch_indices,
      reinforce_indices = _.uniq(_.concat(unpack(reinforce_indices)))
   }
   return cstate
end

function SwitchNode:_evaluate(cstate)
   local routes = self._gater:evaluate(cstate)
   
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
