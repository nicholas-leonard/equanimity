------------------------------------------------------------------------
--[[ SwitchLayer ]]--
-- A layer of parallel SwitchNodes
------------------------------------------------------------------------
local SwitchLayer, parent = torch.class("dp.SwitchLayer", "dp.Container")
SwitchLayer.isSwitchLayer = true

function SwitchLayer:__init(config)
   config = config or {}
   local args, nodes = xlux.unpack(
      {config},
      'SwitchLayer', nil,
      {arg='nodes', type='table', req=true}
   )
   config.typename = 'switchlayer'
   parent.__init(self, config)
   self._nodes = nodes
   self._models = nodes
end

function SwitchLayer:setup(config)
   parent.setup(self, config)
   config.container = self
   config.predecessor = self.predecessor
   config.successor = self.successor
   for i, model in ipairs(self._models) do
      config.id = self:id():create('s'..i)
      model:setup(config)
   end
   self._data_view = self._models[1]:dataView()
end

function SwitchLayer:forward(state)
   local input = state.input
   if input.act then 
      state.input = {input}
   end
   local carry = state.carry
   assert(carry)
   if carry.batch_indices then
      state.carry = {carry}
   end
   return parent.forward(self, state)
end

function SwitchLayer:_forward(input_cstates)
   local output_cstates = {}
   self.ostate = {}
   -- Concatenate node output states into this layer's ostate.
   -- This abstracts the tree structure from the next layer.
   -- Note that not all node branches (experts) will have ostates.
   for input_idx,istate in pairs(self.istate) do
      local node = self._nodes[input_idx]
      local nExperts = node:nExperts()
      local state = {
         input=istate,
         global=self.global,
         carry=input_cstates[input_idx]
      }
      local ostates, cstates = node:forward(state)
      for output_idx, ostate in pairs(ostates) do
         local layer_idx = (input_idx-1) * nExperts + output_idx
         self.ostate[layer_idx] = ostate
         output_cstates[layer_idx] = cstates[output_idx]
      end
   end
   return output_cstates
end

function SwitchLayer:backward(state)
   assert(state.carry)
   parent.backward(self, state)
end

function SwitchLayer:_backward(output_cstates)
   local cstates = {}
   for input_idx, istate in pairs(self.istate) do
      local node = self._nodes[input_idx]
      local nExperts = node:nExperts()
      local start = (input_idx -1) * nExperts
      local stop = start + nExperts
      local cstate = {}
      local ostate = {}
      local output_idx = 1
      for layer_idx = start,stop do
         cstate[output_idx] = output_cstates[layer_idx]
         ostate[output_idx] = self.ostates[layer_idx]
         output_idx = output_idx + 1
      end
      local state = {
         global=self.global,
         carry=cstates[input_idx],
         output=ostate
      }
      self.istates[input_idx], cstates[input_idx] = node:backward(state)
   end
   return cstates
end
