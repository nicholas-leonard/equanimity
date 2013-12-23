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

function SwitchNode:size()
   return #self._experts
end

function SwitchNode:get(index)
   return self._experts[index]
end


function SwithLayer:forward(state)
   local input = state.input
   if input.act then 
      state.input = {input}
   end
   parent.forward(self, state)
end

function SwitchNode:_forward(gstate)
   -- forward gater to get routes
   local gater_ostate = self._gater:forward{global=state}
   -- routes is a matrix of indices
   local routes = gater_ostate.routes
   -- alphas is a matrix of rows that some to one and weight
   local alphas = gater_ostate.alphas
   -- multiply these alphas by any previous alphas
   if self.istate.alphas then
      alphas:cmul(self.istate.alphas)
   end
   local input_act = self.istate.act
   local input_indices = self.istate.indices
   local batch_indices = {}
   local experts = {}
   -- TODO modify multinomial to return what the following loop returns
   -- accumulate example indices and alphas for each experts
   for example_idx = 1,routes:size(1) do
      for sample_idx = 1,routes:size(2) do
         local expert_idx = routes{example_idx,sample_idx}
         local expert = experts[expert_idx] or {examples={}, alphas={}}
         table.insert(expert.examples, example_idx)
         table.insert(expert.alphas, alphas{example_idx,expert_idx})
         experts[expert_idx] = expert
      end
   end   
   self.ostate = {gater=gater_ostate}
   for expert_idx, expert_branch in ipairs(self._experts) do
      local expert = experts[expert_idx]
      if examples then
         -- create a tensor-batch examples for the expert
         local indices = torch.LongTensor(expert.examples)
         expert_branch.istate = {
            batch_indices = indices,
            act = input_act:index(1, indices),
            indices = input_indices:index(1, indices)
         }      
         -- forward batch and update i/o states
         local expert_ostate = expert_branch:forward{global=gstate}
         -- save the alphas
         expert_ostate.alphas 
            = torch.DoubleTensor(expert.alphas):type(self:type())
         self.ostate[expert_idx] = expert_ostate
      else
         -- is this really necessary?
         expert.istate = {}
         expert.ostate = {}
      end
   end
end

function SwitchNode:nExperts()
   return #self._experts
end

function SwitchNode:_evaluate(gstate)
   local routes = self._gater:evaluate(gstate)
   
end

function SwitchNode:_backward(gstate, scale)
   scale = scale or 1
   -- backward gater to get routes
   local gater_istate = self._gater:backward(gstate)
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
------------------------------------------------------------------------
local SwitchLayer, parent = torch.class("dp.SwitchLayer", "dp.Container")
TreeLayer.isTreeLayer

function SwitchLayer:__init(config)
   config = config or {}
   local args, nodes = xlux.unpack(
      {config},
      'SwitchLayer',nil,
      {arg='nodes', type='table', req=true}
   )
   config.typename = 'switchlayer'
   parent.__init(self, config)
   self._nodes = nodes
   self._models = nodes
end

function SwitchLayer:setup(config)
   
end

function SwithLayer:forward(state)
   local input = state.input
   if input.act then 
      state.input = {input}
   end
   parent.forward(self, state)
end

function SwitchLayer:_forward(gstate)
   self.ostate = {}
   -- Concatenate node output states into this layer's ostate.
   -- This abstract the tree structure from the next layer.
   -- Note that not all node branches (experts) will have ostates.
   for input_idx,node_istate in pairs(self.istate) do
      local node = self._nodes[input_idx]
      local nExperts = node:nExperts()
      for output_idx,node_ostate in pairs(node:forward(node_istate)) do
         self.ostate[(input_idx-1)*nExperts + output_idx] = node_ostate
      end
   end
end

function SwitchLayer:_backward(gstate)

end
