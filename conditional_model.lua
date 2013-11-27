--TODO build conditional model components.
-- refresh memory on paper.
-- will need special criteria
-- will need special backpropagated gradients!!!


------------------------------------------------------------------------
--[[ ConditionalNode ]]--
------------------------------------------------------------------------
local ConditionalNode = torch.class("dp.ConditionalNode", "dp.Model")

function ConditionalNode:acceptForward(input, visitor)
   self._state.inputs = inputs
   visitor:model_preForward(self)
   -- routing_table is a binary matrix or matrix of indexes?
   local routing_table = self._gater:acceptForward(input, visitor)
   local batch_indices = {}
   local input_table = {}
   for i=1,#self._experts do
      -- column in routing table is associated to an expert
      batch_indices[i] = routing_table:sub(2, i)
      -- get examples to be propagated to expert
      input_table[i] = input:index(1, batch_indices[i])
      -- propagate batch to expert
      self._experts[i]:acceptForward(input_table[i], visitor)
   end
   -- Will need this for backward
   self._state.routing_table = routing_table
   --...
   visitor:model_postForward(self)
end

function ConditionalNode:acceptBackward(


------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Propagator")

function Conditioner:__init()
   
end
