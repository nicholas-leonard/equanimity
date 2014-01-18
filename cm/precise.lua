------------------------------------------------------------------------
--[[ Precise ]]--
-- samples routes from multinomial
-- reinforces winning routes using MSE with error distribution as targets
-- zero the grads of unsampled experts
-- epsilon-greedy exploration (TODO, moving distribution)
-- affine transform followed by non-linearity (transfer function)
-- uses sigmoid transfer function by default for inputs to multinomial
------------------------------------------------------------------------
local Precise, parent = torch.class("dp.Precise", "dp.Equanimous")
Precise.isPrecise = true

--E-greedy : http://junedmunshi.wordpress.com/tag/e-greedy-policy/
function Precise:__init(config)
   config = config or {}
   local args, transfer, criterion
      = xlua.unpack(
      {config},
      'Precise', nil,
      {arg='transfer', type='nn.Module', default=nn.LogSoftMax(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='criterion', type='nn.Criterion', 
       default=nn.DistKLDivCriterion()}
   )
   config.typename = config.typename or 'precise'
   config.transfer = transfer
   config.criterion = criterion
   config.targets = {0,1}
   parent.__init(self, config)   
end 

function Precise:_backward(cstate)
   self._gater_error = self._gater_error + self._criterion:forward(
      self.ostate.act_double, self.ostate.gater_targets
   )
   local grad = self._criterion:backward(
      self.ostate.act_double, self.ostate.gater_targets
   )
   self.ostate.grad = grad:type(self.ostate.act:type())
   dp.Neural._backward(self, cstate)
end

function Precise:report()
   local report = parent.report(self)
   local gater_error = self._gater_error/(self._sample_count+0.00001)
   print(self:id():toString()..':loss', gater_error)
   return table.merge(
      report, {loss=gater_error}
   )
end

function Precise:zeroStatistics()
   parent.zeroStatistics(self)
   self._gater_error = 0
end
