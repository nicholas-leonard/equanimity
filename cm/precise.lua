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
      {arg='transfer', type='nn.Module', default=nn.Sigmoid(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='criterion', type='nn.Criterion', 
       default=nn.MSECriterion()}
   )
   config.typename = config.typename or 'precise'
   config.transfer = nn.SoftMax()
   config.criterion = criterion
   config.targets = {0.1,0.9}
   parent.__init(self, config)   
   self._sample_targets = torch.DoubleTensor()
   self._target_dist = torch.exp(torch.range(1,self._n_reinforce)):sort(1,true)
   self._target_dist:div(self._target_dist:max())
   self._target_dist:mul(self._targets[2]-self._targets[1]):add(self._targets[1])
   print(self._target_dist)
end 

function Precise:_backward(cstate)
   local gater_targets = self.ostate.gater_targets:clone()
   --print(self:id():toString(), gater_targets:mean(1))
   gater_targets:add(-gater_targets:min(2):expandAs(gater_targets))
   gater_targets:add(gater_targets:mean(2):mul(self._epsilon):expandAs(gater_targets))
   gater_targets:cdiv(gater_targets:sum(2):expandAs(gater_targets))
   --print(self:id():toString(), gater_targets:mean(1))
   gater_targets = dp.multinomial(gater_targets, self._n_reinforce, true)
   --print(self:id():toString(), self._gater_error/self._sample_count)
   --print(self:id():toString(), gater_targets[{{1,3},{}}])
   self._sample_targets:resizeAs(self.ostate.act_double):fill(self._targets[1])
   for i = 1, gater_targets:size(1) do
      self._sample_targets:select(1,i):indexCopy(1, gater_targets:select(1,i), self._target_dist)
   end
   self._gater_error = self._gater_error + self._criterion:forward(
      self.ostate.act_double, self._sample_targets
   )
   local grad = self._criterion:backward(
      self.ostate.act_double, self._sample_targets
   )
   --print(self:id():toString(), self._grad:mean(1))
   --gater_targets:cdiv(gater_targets:sum(1):expandAs(gater_targets))
   --local n_equanimous = self._n_reinforce * gater_targets:size(1) / gater_targets:size(2)
  -- gater_targets = dp.multinomial(gater_targets:t(), , true):reshape(gater_targets:size(1))
   self.ostate.grad = grad:type(self.ostate.act:type())
   dp.Neural._backward(self, cstate)
end

function Precise:report()
   local report = parent.report(self)
   local gater_error = self._gater_error/(self._sample_count+0.00001)
   print(self:id():toString()..':loss', gater_error, self._sample_count)
   return table.merge(
      report, {loss=gater_error}
   )
end

function Precise:zeroStatistics()
   parent.zeroStatistics(self)
   self._gater_error = 0
end
