------------------------------------------------------------------------
--[[ RouterNode ]]--
-- a routed node in a tree.
-- branches are experts.
-- each example gets routed through different branches by the gater.
-- only differs from SwitchNode in _backward()
------------------------------------------------------------------------
local RouterNode, parent = torch.class("dp.RouterNode", "dp.SwitchNode")
RouterNode.isRouterNode = true

function RouterNode:__init(config)
   config = config or {}
   config.typename = 'router'
   parent.__init(self, config)
end

function RouterNode:_backward(cstates)
   local istate = {}
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
      local expert_istate = expert_branch:backward{
         output=expert_ostate, global=self.global, carry=expert_cstate
      }
      self._expert_grad:zero()
      self._expert_grad:indexCopy(1, expert_indices, expert_istate.grad:double())
      -- accumulate input gradients from experts by addition
      self._input_grad:add(self._expert_grad:type(self._input_grad:type()))
   end 
   --[[ gater ]]--
   local gater_istate = self._gater:backward{
      global=self.gstate, carry={scale=1/n_example}
   }
   self.istate.grad = self._input_grad
   local cstate = {batch_indices = self.istate.batch_indices}
   return cstate
end
