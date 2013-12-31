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
         epsilon, eval_proto
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
       help='evaluation protocol : MAP, Stochastic Sampling, etc'}
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
   -- statistics :
   --- sparsity : 
   ---- distribution of alphas in bins
   local n_bin = self._output_size
   self._alpha_bins 
      = torch.range(1,n_bin):double():div(n_bin):storage():totable()
   self._alpha_dist = torch.DoubleTensor(n_bin)
   self:zeroStatistics()
end

function Equanimous:setup(config)
   parent.setup(self, config)
end   

function Equanimous:_forward(cstate)
   -- affine transform + transfer function
   parent._forward(self, cstate)
   self.ostate.act_double = self.ostate.act:double()
   -- e-greedy on entire batch
   local p = math.random()
   -- alphas are used to weigh a mean of leaf outputs
   local alphas
   local e_greedy = false
   if p > self._epsilon then
      alphas = torch.add(self.ostate.act_double, -self._targets[1])
      alphas:cdiv(alphas:sum(2):expandAs(alphas))
   else
      e_greedy = true
      alphas = self._alpha_dist:clone()
      -- reverse distribution and make unlikely alphas more likely
      alphas:add(-alphas:max()):mul(-1):div(alphas:sum())
      alphas = alphas:reshape(1,alphas:size(1)):expandAs(self.ostate.act_double)
   end
   self.ostate.alphas = alphas
   -- sample from a multinomial without replacement
   self.ostate.routes = dp.multinomial(
      self.ostate.alphas, self._n_sample, true
   )
   -- gather stats
   if not e_greedy then
      local previous = 0
      for i,upper in ipairs(self._alpha_bins) do
         local current = torch.le(alphas,upper):double():sum()
         self._alpha_dist[i] = self._alpha_dist[i] + current - previous
         previous = current
      end
   end
   self._sample_count = self._sample_count + alphas:size(1)
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
   print(dr, self._sample_count)
   return {
      alpha = dist_report,
      n_sample = self._sample_count
   }
end

function Equanimous:zeroStatistics()
   self._alpha_dist:zero()
   self._sample_count = 0
end

