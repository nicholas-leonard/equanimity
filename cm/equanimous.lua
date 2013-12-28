------------------------------------------------------------------------
--[[ Equanimous ]]--
-- samples routes from multinomial
-- reinforces winning routes using MSE with 0/1 targets
-- epsilon-greedy exploration
-- affine transform followed by non-linearity (transfer function)
-- uses sigmoid transfer function by default for inputs to multinomial
------------------------------------------------------------------------
local Equanimous, parent = torch.class("dp.Equanimous", "dp.Neural")
Equanimous.isEquanimous = true

--E-greedy : http://junedmunshi.wordpress.com/tag/e-greedy-policy/
function Equanimous:__init(config)
   config = config or {}
   local args, n_sample, n_reinforce, n_eval, targets, transfer, epsilon 
      = xlua.unpack(
      {config},
      'Equanimous', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example during training.'},
      {arg='n_reinforce', type='number', 
       help='number of experts reinforced per example during training.'},
      {arg='n_eval', type='number', 
       help='number of experts chosen per example during evaluation.'},
      {arg='targets', type='table', default={0.1,0.9},
       help='targets used to diminish (first) and reinforce (last)'},
      {arg='transfer', type='nn.Module', default=nn.Sigmoid(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='epsilon', type='number', default=0.1,
       help='probability of sampling from uniform instead of multinomial'}
   )
   config.typename = config.typename or 'equanimous'
   config.transfer = transfer
   parent.__init(self, config)   
   self._n_sample = n_sample
   self._n_reinforce = n_reinforce
   self._n_eval = n_eval
   self._targets = targets
   self._epsilon = epsilon
   self._criterion = nn.MSECriterion()
end

function Equanimous:setup(config)
   parent.setup(self, config)
end   

function Equanimous:_forward(cstate)
   -- affine transform + transfer function
   parent._forward(self, cstate)
   -- sample n_sampe samples from a multinomial without replacement
   local n_sample = self._n_sample
   self.ostate.act_double = self.ostate.act:double()
   self.ostate.routes = dp.multinomial(
      self.ostate.act_double, n_sample, true
   )
   -- alphas will eventually be used to weigh a weighted mean of outputs
   self.ostate.alphas = torch.add(
      self.ostate.act_double, -self._targets[1]
   )
end

function Equanimous:_backward(cstate, scale)
   local targets = self.ostate.act_double:clone():fill(self._targets[1])
   -- TODO :
   -- compare to alternative iteration over examples (batch_idx)
   for expert_idx, reinforce in pairs(self.ostate.reinforce_indices) do
      targets[{{},expert_idx}]:indexFill(1, reinforce, self._targets[2])
   end
   self.ostate.grad = self._criterion:backward(
      self.ostate.act_double, targets
   ):type(self.ostate.act:type())
   parent._backward(self, cstate)
end

function Equanimous:_evaluate(cstate)
   if self._evaluate_protocol == 'MAP' then
      self._evaluateMap(gstate)
   end
end

function Equanimous:report()

end

function Equanimous:_evaluateMAP(gstate)
   
end
