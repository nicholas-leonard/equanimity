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
   local args, n_sample, n_reinforce, targets, transfer, 
         epsilon, eval_proto, lambda, ema
      = xlua.unpack(
      {config},
      'Equanimous', nil,
      {arg='n_sample', type='number', 
       help='number of experts sampled per example'},
      {arg='n_reinforce', type='number', 
       help='number of experts reinforced per example during training.'},
      {arg='targets', type='table', default={0.1,0.9},
       help='targets used to diminish (first) and reinforce (last)'},
      {arg='transfer', type='nn.Module', default=nn.Sigmoid(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='epsilon', type='number', default=0.1,
       help='probability of sampling from inverse distribution'},
      {arg='eval_proto', type='string', default='MAP',
       help='evaluation protocol : MAP, Stochastic Sampling, etc'},
      --[[{arg='n_focus', type='number', 
       help='nb of experts to focus on defaults to n_sample'},--]]
      {arg='lambda', type='number', default=0, 
       help='weight of inverse marginal expert multinomial dist'},
      {arg='ema', type='number', default=0.5,
       help='weight of present for computing exponential moving avg'}
   )
   config.typename = config.typename or 'equanimous'
   config.transfer = transfer
   parent.__init(self, config)   
   self._n_sample = n_sample
   self._n_reinforce = n_reinforce
   self._targets = targets
   self._epsilon = epsilon
   self._eval_proto = eval_proto
   self._criterion = nn.MSECriterion()
   self._criterion.sizeAverage = false
   self._lambda = lambda
   self._ema = ema
   -- statistics :
   --- sparsity : 
   ---- distribution of alphas in bins
   local n_bin = self._output_size
   self._alpha_bins 
      = torch.range(1,n_bin):double():div(n_bin):storage():totable()
   self._alpha_dist = torch.DoubleTensor(n_bin)
   self._ema_dist = torch.DoubleTensor(n_bin):zero()
   self:zeroStatistics()
end

function Equanimous:setup(config)
   parent.setup(self, config)
end   

function Equanimous:_forward(cstate)
   if self.gstate.focus == 'examples' then
      self:_exampleFocus(cstate)
   elseif self.gstate.focus == 'experts' then
      self:_expertFocus(cstate)
   end
end

function Equanimous:_exampleFocus(cstate)
   local start = os.clock()
   -- affine transform + transfer function
   parent._forward(self, cstate)
   self.ostate.act_double = self.ostate.act:double()
   -- e-greedy on entire batch
   local p = math.random()
   -- alphas are used to weigh a mean of leaf outputs
   local alphas
   local e_greedy = false
   if p >= self._epsilon then
      alphas = torch.add(self.ostate.act_double, -self._targets[1])
      -- normalize each example's alphas to sum to one
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   else
      e_greedy = true
      alphas = dp.reverseDist(self._ema_dist):reshape(
         1, self._ema_dist:size(1)
      ):expandAs(self.ostate.act_double)
   end
   self.ostate.alphas = alphas
   -- equanimity bias
   local biased_alphas = alphas
   if self._lambda > 0 then
      biased_alphas = torch.add(
         torch.mul(alphas, 1-self._lambda), 
         torch.mul(
            dp.reverseDist(self._ema_dist), self._lambda
         ):reshape(1,self._ema_dist:size(1)):expandAs(alphas)
      )
   end
   -- sample experts from an example multinomial without replacement
   self.ostate.routes = dp.multinomial(
      biased_alphas, self._n_sample, true
   )
   if DEBUG then
      print(e_greedy)
      print("activations")
      print(self.ostate.act_double)
      print("alphas")
      print(alphas)
      print("routes")
      print(self.ostate.routes)
   end
   -- gather stats
   if not e_greedy then
      local previous = 0
      for i,upper in ipairs(self._alpha_bins) do
         local current = torch.le(alphas,upper):double():sum()
         self._alpha_dist[i] = self._alpha_dist[i] + current - previous
         previous = current
      end
      -- exponential moving average of alphas
      local current_mean = alphas:mean(1)
      self._ema_dist = self._ema_dist:mul(1-self._ema)
         + current_mean:div(current_mean:sum()):mul(self._ema)
   end
   self._sample_count = self._sample_count + alphas:size(1)
   --print("example", os.clock()-start)
end

function Equanimous:_expertFocus(cstate)
   local start = os.clock()
   -- affine transform + transfer function
   parent._forward(self, cstate)
   self.ostate.act_double = self.ostate.act:double()
   -- reverse distribution and make unlikely experts more likely
   local expert_dist = dp.reverseDist(self._ema_dist)
   expert_dist:mul(self._n_sample * self.ostate.act_double:size(1))
   
   -- alphas are used to weigh a mean of leaf outputs
   local alphas = torch.add(self.ostate.act_double, -self._targets[1])
   -- normalize each examples's alphas to sum to one
   alphas:cdiv(alphas:sum(2):expandAs(alphas))
   self.ostate.alphas = alphas
   -- normalize each expert's alphas to sum to one to get probs
   local probs = torch.cdiv(alphas, alphas:sum(1):expandAs(alphas))
   local experts = {}
   local start2 = os.clock()
   for expert_idx = 1,expert_dist:size(1) do
      -- sample examples from an expert multinomial without replacement
      local expert_indices = dp.multinomial(
         probs:select(2, expert_idx), 
         math.max(2, math.ceil(expert_dist[expert_idx])), 
         false
      )
      experts[expert_idx] = {expert_indices = expert_indices}
   end
   --print("example2", os.clock()-start2)
   if DEBUG then
      print("activation")
      print(self.ostate.act_double)
      print("alphas")
      print(alphas)
      print("probs")
      print(probs)
      print"expert_dist"
      print(expert_dist)
   end
   self.ostate.experts = experts
   --print("example", os.clock()-start)
end

function Equanimous:_backward(cstate, scale)
   local targets = self.ostate.act_double:clone():fill(self._targets[1])
   -- TODO : compare to alternative iteration over examples (batch_idx)
   for expert_idx, reinforce in pairs(self.ostate.reinforce_indices) do
      targets[{{},expert_idx}]:indexFill(1, reinforce, self._targets[2])
   end
   self.ostate.grad = self._criterion:backward(
      self.ostate.act_double, targets
   ):type(self.ostate.act:type())
   parent._backward(self, cstate)
end

function Equanimous:_evaluate(cstate)
   parent._forward(self, cstate)
   self.ostate.act_double = self.ostate.act:double()
   if self._eval_proto == 'MAP' then
      return self._evaluateMAP(self, cstate)
   end
   error"NotImplemented"
end

function Equanimous:_evaluateMAP(cstate)
   local alphas = torch.add(self.ostate.act_double, -self._targets[1])
   alphas:cdiv(alphas:sum(2):expandAs(alphas))
   self.ostate.alphas = alphas
   local __, routes = torch.sort(alphas, 2, true)
   self.ostate.routes = routes[{{},{1,self._n_sample}}]
end

function Equanimous:report()
   local dist_report = dp.distReport(self._alpha_dist)
   dist_report.bins = self._alpha_bins
   local dr = table.copy(dist_report)
   dr.dist = table.tostring(dr.dist:storage():totable())
   dr.bins = table.tostring(dr.bins)
   dr.name = self:id():toString()
   print(dr.dist)
   return {
      alpha = dist_report,
      n_sample = self._sample_count
   }
end

function Equanimous:zeroStatistics()
   self._alpha_dist:zero()
   self._sample_count = 0
end

