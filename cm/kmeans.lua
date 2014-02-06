------------------------------------------------------------------------
--[[ Kmeans ]]--
-- An unsupervised k-means clustering module
------------------------------------------------------------------------
local Kmeans, parent = torch.class("dp.Kmeans", "dp.Model")
Kmeans.isKmeans = true

function Kmeans:__init(config)
   config = config or {}
   local tags = {
      ['no-momentum']=true, 
      ['no-maxnorm']=true, 
      ['no-weightdecay']=true
   }
   local args, input_size, k, n_sample, sim_proto, typename, tags 
      = xlua.unpack(
      {config},
      'Kmeans', 
      'An unsupervised k-means clustering module',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='k', type='number', req=true, help='Number of centroids'},
      {arg='n_sample', type='number', default=1,
       help='number of nearest centroids to sample per example'},
      {arg='sim_proto', type='string', default='rev_dist',
       help='similarity protocol.'},
      {arg='typename', type='string', default='kmeans', 
       help='identifies Model type in reports.'},
      {arg='tags', type='table', default=tags}
   )
   config.typename = typename
   config.tags = tags
   parent.__init(self, config)
   self._k = k
   self._input_size = input_size
   self._n_sample = n_sample
   self._sim_proto = sim_proto
   self._tags.hasParams = true
   -- tensors
   self._centroids = torch.Tensor(self._k, self._input_size)
   self._D = torch.Tensor()
   self._gradCentroids = torch.Tensor(self._k, self._input_size):zero()
   self._totalcounts = torch.Tensor(self._k)
   self:reset()
   self:zeroStatistics()
end

function Kmeans:setup(config)
   parent.setup(self, config)
   self._data_view = 'feature'
end

function Kmeans:_forward(cstate)
   local x = self.istate.act

   local x2 = x:clone():pow(2):sum(2):mul(0.5)
   -- sums of squares
   local c2 = self._centroids:clone():pow(2):sum(2):mul(0.5)

   -- process batch
   -- indices
   local x_t = x:t()
   -- k X batch_size
   self._D:resize(self._k, x_t:size(2)):zero()
   self._D:addmm(self._centroids, x_t)
   self._D:mul(-1)
   self._D:add(c2:resize(c2:size(1), 1):expandAs(self._D))
   self._D:add(x2:resize(1, x2:size(1)):expandAs(self._D))
   -- minimum distance : k x batch_size
   local distances,labels = torch.sort(self._D:double(),1)
   self._loss = self._loss + distances:select(1,1):sum()
   
   -- count examples per centroid
   local S = torch.zeros(x:size(1), self._k)
   local winners = labels:select(1,1)
   for i = 1,winners:size(1) do
      S[{i,winners[i]}] = 1
   end
   
   -- activation is just nearest centroid (one-hot encoding)
   self.ostate.act = S:type(self.istate.act:type())
   self:_similarity(self._D:t(), labels:t())
   return cstate
end

function Kmeans:_similarity(distances, labels)
   distances = distances:type(self.istate.act:type())
   local similarity
   if self._sim_proto == 'rev_dist' then
      distances:add(-1, distances:min(2):expandAs(distances))
      distances:cdiv(distances:sum(2):resize(distances:size(1),1):expandAs(distances))
      similarity = dp.reverseDist(distances, true)
   elseif self._sim_proto == 'uniform' then
      distances:ones():div(distances:size(2))
      similarity = distances
   else
      error"unknown similarity protocol"
   end
   --if _.isNaN(similarity:sum()) then print"NaN1"; os.exit() else print("AN", similarity:size()) end
   --print(_.isNaN(self._D:sum()))
   self.ostate.alphas = similarity:double()
   self.ostate.routes = labels[{{},{1,self._n_sample}}]
end

function Kmeans:_backward(cstate)
   -- sum of examples in each centroid : k x input_size 
   self._gradCentroids:addmm(self.ostate.act:t(), self.istate.act)
   -- counts of examples in each centroid : k
   local counts = self.ostate.act:sum(1):add(0.000001)
   
   -- mean of examples in centroid = sum/count
   self._gradCentroids:cdiv(counts:resize(self._k, 1):expandAs(self._gradCentroids))
   -- vectors between centroids and mean of examples
   self._gradCentroids:add(-1, self._centroids):mul(-1)
   -- total counts
   self._totalcounts:add(counts)
   self._sample_count = self._sample_count + self.istate.act:size(1)
   return cstate
end

function Kmeans:type(type)
   self._centroids = self._centroids:type(type)
   self._totalcounts = self._totalcounts:type(type)
   self._D = self._D:type(type)
   self._gradCentroids = self._gradCentroids:type(type)
end

function Kmeans:reset()
   self._centroids:copy(torch.rand(self._k,self._input_size))
   self._totalcounts:zero()
end

function Kmeans:parameters()
   local params = {}
   if not params.weight then
      params.weight = {param = self._centroids, grad=self._gradCentroids}
   end
   return params
end

function Kmeans:maxNorm(max_out_norm, max_in_norm)
   if not self.backwarded then return end
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = params.weight.param
   if max_out_norm then
      -- rows feed into output neurons 
      dp.constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      dp.constrain_norms(max_in_norm, 1, weight)
   end
end

function Kmeans:report()
   local report = parent.report(self) or {}
   report.loss = self._loss / self._sample_count
   report.totalcounts = self._totalcounts:clone():double()
   print(self:name(), report.loss, table.tostring(report.totalcounts:storage():totable()))
   return report
end

function Kmeans:zeroStatistics()
   parent.zeroStatistics(self)
   self._totalcounts:zero()
   self._sample_count = 0
   self._loss = 0
end

function Kmeans:doneEpoch(report, ...)
   -- reset dead clusters
   local totalcounts = self._totalcounts:clone():double()
   local min_count = totalcounts:mean()*0.1
   local dead = totalcounts:lt(min_count)
   local __, dead_indices = totalcounts:sort()
   local dead_count = dead:double():sum()
   if dead_count > 0 then
      totalcounts[dead] = 0
      -- sample from most populous clusters
      print(totalcounts, dead_count)
      local archtype_indices = dp.multinomial(totalcounts, dead_count)
      local archtypes = self._centroids:double():index(1, archtype_indices)
      print(archtype_indices, archtypes, dead_indices[{{1,dead_count}}])
      -- small variations
      archtypes:add(torch.randn(archtypes:size()):mul(0.001))
      local centroids = self._centroids:double()
      centroids:indexCopy(1, dead_indices[{{1,dead_count}}], archtypes)
      self._centroids:copy(centroids)
   end
   -- zeros statistics
   self:zeroStatistics()
end
