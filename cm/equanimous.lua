------------------------------------------------------------------------
--[[ Equanimous ]]--
-- samples routes from multinomial
-- reinforces winning routes using MSE with 0/1 targets
-- epsilon-greedy exploration (TODO, moving distribution)
-- affine transform followed by non-linearity (transfer function)
-- uses sigmoid transfer function by default for inputs to multinomial
------------------------------------------------------------------------
local Equanimous, parent = torch.class("dp.Equanimous", "dp.Neural")
Equanimous.isEquanimous = true

--E-greedy : http://junedmunshi.wordpress.com/tag/e-greedy-policy/
function Equanimous:__init(config)
   config = config or {}
   require 'nnx'
   local args, n_sample, targets, transfer, epsilon, epsilon_decay,
      temperature, temperature_decay, eval_proto, criterion
      = xlua.unpack(
      {config},
      'Equanimous', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example'},
      {arg='targets', type='table', default={0.1,0.9},
       help='targets used to diminish (first) and reinforce (last)'},
      {arg='transfer', type='nn.Module', default=nn.Identity(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='epsilon', type='number', default=0.1,
       help='probability of sampling from inverse distribution'},
      {arg='epsilon_decay', type='number', default=0.99,
       help='epsilon is decayed by this factor every epoch'}, 
      {arg='temperature', type='number', default=1,
       help='high temperature means exploration, low means exploitation'},
      {arg='temperature_decay', type='number', default=0.99,
       help='temperature is decayed by this factor every epoch'},
      {arg='eval_proto', type='string', default='MAP',
       help='evaluation protocol : MAP, Stochastic Sampling, etc'},
      {arg='criterion', type='nn.Criterion', 
       default=nn.DistNLLCriterion{inputIsProbability=true},
       help='Criterion to be used for optimizing winning experts'}
   )
   config.typename = config.typename or 'equanimous'
   config.transfer = transfer
   parent.__init(self, config)  
   self._n_sample = n_sample
   self._targets = targets
   self._epsilon = epsilon
   self._temperature = temperature
   self._epsilon_decay = epsilon_decay
   self._temperature_decay = temperature_decay
   self._eval_proto = eval_proto
   self._criterion = criterion
   self._criterion.sizeAverage = false
   -- statistics : sparsity distribution of alphas
   local n_bin = 10
   self._alpha_bins = torch.range(1,n_bin):double():div(n_bin):storage():totable()
   self._alpha_dist = torch.DoubleTensor(n_bin)
   self:parameters().bias.param:zero()
   self:zeroStatistics()
   self._softmax = nn.SoftMax()
end

function Equanimous:setup(config)
   parent.setup(self, config)
end   

function Equanimous:_forward(cstate)
   -- affine transform + transfer function
   parent._forward(self, cstate)
   -- temperature increases exploration by making multinomial uniform
   self.ostate.pre_softmax = self.ostate.act:double():div(self._temperature)
   self.ostate.act_double = self._softmax:forward(self.ostate.pre_softmax)
   -- alphas are used to weigh a mean of leaf outputs
   if not self._alphas then
      self._alphas = self.ostate.act_double:clone()
   end
   self._alphas:resize(self.ostate.act:size())
   --self._alphas:copy(self.ostate.act):add(-self._targets[1])
   self._alphas:copy(self.ostate.act_double)
   -- high epsilon = exploration
   if self._epsilon > 0 then
      self._alphas:add(self._alphas:mean(2):mul(self._epsilon):expandAs(self._alphas))
      self._alphas:cdiv(self._alphas:sum(2):add(0.000001):expandAs(self._alphas))
   end
   --[[for i = 1,3 do
      -- Equanimity : normalize each expert's alphas to sum to one 
      self._alphas:cdiv(self._alphas:sum(1):expandAs(self._alphas))
      -- normalize each example's alphas to sum to one
      self._alphas:cdiv(self._alphas:sum(2):expandAs(self._alphas))
   end--]]
   self.ostate.alphas = self._alphas
   -- sample experts from an example multinomial without replacement
   self.ostate.routes = dp.multinomial(self._alphas, self._n_sample, true)
   -- gather stats
   local previous = 0
   local alphas = self._alphas
   for i,upper in ipairs(self._alpha_bins) do
      local current = torch.le(alphas,upper):double():sum()
      self._alpha_dist[i] = self._alpha_dist[i] + current - previous
      previous = current
   end
   self._sample_count = self._sample_count + alphas:size(1)
end

function Equanimous:_backward(cstate)
   self.ostate.act_double:add(0.00001)
   self.ostate.gater_targets:add(0.00001)
   self._gater_error = self._gater_error + self._criterion:forward(
      self.ostate.act_double, self.ostate.gater_targets
   )
   local grad = self._criterion:backward(
      self.ostate.act_double, self.ostate.gater_targets
   )
   grad = self._softmax:backward(
      self.ostate.pre_softmax, grad
   )--:mul(self._temperature)
   self.ostate.grad = grad:type(self.ostate.act:type())
   parent._backward(self, cstate)
end

function Equanimous:_evaluate(cstate)
   parent._forward(self, cstate)
   -- temperature increases exploration by making multinomial uniform
   self.ostate.act_double = self._softmax:forward(
      self.ostate.act:double():div(self._temperature)
   )
   if self._eval_proto == 'MAP' then
      return self._evaluateMAP(self, cstate)
   elseif self._eval_proto == 'EMAP' then
      return self._evaluateEMAP(self, cstate)
   elseif self._eval_proto == 'AMAP' then
      return self._evaluateAMAP(self, cstate)
   elseif self._eval_proto == 'Random' then
      return self._evaluateRandom(self, cstate)
   end
   error"NotImplemented"
end

function Equanimous:_evaluateMAP(cstate)
   local alphas = torch.add(
      self.ostate.act_double, ---self._targets[1]
      -self.ostate.act_double:min(2):expandAs(self.ostate.act_double)
   )
   alphas:cdiv(alphas:sum(2):add(0.00001):expandAs(alphas))
   self.ostate.alphas = alphas
   local __, routes = torch.sort(alphas, 2, true)
   self.ostate.routes = routes[{{},{1,self._n_sample}}]
end

function Equanimous:_evaluateAMAP(cstate)
   local alphas = torch.add(
      self.ostate.act_double, ---self._targets[1]
      -self.ostate.act_double:min(2):expandAs(self.ostate.act_double)
   )
   alphas:cdiv(alphas:sum(2):add(0.00001):expandAs(alphas))
   self.ostate.alphas = dp.reverseDist(alphas)
   local __, routes = torch.sort(alphas, 2, true)
   self.ostate.routes = routes[{{},{1,self._n_sample}}]
end

function Equanimous:_evaluateRandom(cstate)
   local alphas = torch.rand(self.ostate.act_double:size())
   alphas:cdiv(alphas:sum(2):add(0.00001):expandAs(alphas))
   self.ostate.alphas = alphas
   local __, routes = torch.sort(alphas, 2, true)
   self.ostate.routes = routes[{{},{1,self._n_sample}}]
end

-- equanimous MAP
function Equanimous:_evaluateEMAP(cstate)
    -- alphas are used to weigh a mean of leaf outputs
   if not self._alphas then
      self._alphas = self.ostate.act:clone()
   end
   self._alphas:resize(self.ostate.act:size()):copy(self.ostate.act):add(-self._targets[1])
   for i = 1,3 do
      -- Equanimity : normalize each expert's alphas to sum to one 
      self._alphas:cdiv(self._alphas:sum(1):expandAs(self._alphas))
      -- normalize each example's alphas to sum to one
      self._alphas:cdiv(self._alphas:sum(2):expandAs(self._alphas))
   end
   self.ostate.alphas = self._alphas:double()
   local __, routes = torch.sort(self.ostate.alphas, 2, true)
   self.ostate.routes = routes[{{},{1,self._n_sample}}]
end

function Equanimous:boundTargets(targets)
   targets:mul(self._targets[2]-self._targets[1]):add(self._targets[1])
end

function Equanimous:report()
   local report = parent.report(self)
   local dist_report = dp.distReport(self._alpha_dist)
   local gater_error = self._gater_error/(self._sample_count+0.00001)
   print(self:id():toString()..':loss', gater_error, self._sample_count)
   return table.merge(
      report, {
         alpha=dist_report, n_sample=self._sample_count, 
         loss=gater_error, epsilon=self._epsilon,
         temperature=self._temperature
      }
   )
end

function Equanimous:zeroStatistics()
   parent.zeroStatistics(self)
   if self._alpha_dist then
      self._alpha_dist:zero()
   end
   self._sample_count = 0
   self._gater_error = 0
end

function Equanimous:doneEpoch(report)
   self._temperature = math.max(1, self._temperature * self._temperature_decay)
   self._epsilon = self._epsilon * self._epsilon_decay
   parent.doneEpoch(self, report)
end

