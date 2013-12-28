------------------------------------------------------------------------
--[[ SwitchLayer ]]--
-- A layer of parallel SwitchNodes
------------------------------------------------------------------------
local SwitchLayer, parent = torch.class("dp.SwitchLayer", "dp.Container")
SwitchLayer.isSwitchLayer = true

function SwitchLayer:__init(config)
   config = config or {}
   local args, nodes = xlua.unpack(
      {config},
      'SwitchLayer', nil,
      {arg='nodes', type='table', default={},
       help='a list of SwitchNodes.'}
   )
   config.typename = 'switchlayer'
   config.models = nodes
   parent.__init(self, config)
   -- just a name change
   self._nodes = self._models
end

function SwitchLayer:setup(config)
   parent.setup(self, config)
   config.container = self
   config.predecessor = self.predecessor
   config.successor = self.successor
   for i, model in ipairs(self._models) do
      config.id = self:id():create('sn'..i)
      model:setup(config)
   end
   self._data_view = self._models[1]:dataView()
end

function SwitchLayer:forward(state)
   -- if tensor, then makes table with tensor as member 'act'
   self:setInputState(state.input)
   -- if an istate from a non-switch models, then listify
   local input = self.istate
   if input.act then 
      state.input = {input}
      self._listified = true
   end
   local carry = state.carry
   -- for batch_indices
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
         input=istate, global=self.global,
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
   -- for reinforce indices
   assert(state.carry)
   local istate, cstate = parent.backward(self, state)
   -- un-listify (for predecessor non-switch models) 
   if self._listified then
      istate = istate[1]
      cstate = cstate[1]
      assert(type(istate) == 'table')
      assert(type(cstate) == 'table')
   end
   return istate, cstate
end

function SwitchLayer:_backward(output_cstates)
   local cstates = {}
   for input_idx, istate in pairs(self.istate) do
      local node = self._nodes[input_idx]
      local nExperts = node:nExperts()
      local start = ((input_idx -1) * nExperts) + 1
      local cstate = {}
      local ostate = {}
      local output_idx = 1
      for layer_idx = start,start + nExperts - 1 do
         cstate[output_idx] = output_cstates[layer_idx]
         ostate[output_idx] = self.ostate[layer_idx]
         output_idx = output_idx + 1
      end
      local state = {
         global=self.global, carry=cstate, output=ostate
      }
      self.istate[input_idx], cstates[input_idx] = node:backward(state)
   end
   return cstates
end
