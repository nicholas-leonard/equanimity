------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Optimizer")

function Conditioner:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample())
   
   local ostates = model:forward{
      input=batch:inputs(), carry={batch_indices=batch_indices},
      global={focus='examples'}
   }
   
   local loss, outputs = self._criterion:forward(
      ostates, batch:targets(), batch_indices
   )
   
   batch:setLoss(loss)  
   batch:setOutputs(outputs)
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneFeedback", 
                          report, batch)
   
   --[[ backpropagate ]]--
   local istates, cstates = self._criterion:backward(
      ostates, batch:targets(), batch:indices()
   )
   model:backward{
      output=istates, carry=cstates, global={focus='examples'}
   }
   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          report, batch)
                          
end

function Conditioner:report()
   local report = parent.report(self)
   report.essrl = self._criterion:report()
   return report
end

function Conditioner:resetLoss()
   parent.resetLoss(self)
   self._criterion:resetStatistics()
end

------------------------------------------------------------------------
--[[ Equanimizer ]]--
------------------------------------------------------------------------
local Equanimizer, parent = torch.class("dp.Equanimizer", "dp.Conditioner")

function Equanimizer:__init(config)
   config = config or {}
   local args, n_leaf = xlua.unpack(
      {config},
      'Equanimizer', 
      'Adds a second training phase to Conditioner. '..
      'It focuses on experts in order to impose an equanimous '..
      'constraint that balances the distribution of expert-examples.',
      {arg='n_leaf', type='number', req=true, 
       help='number of leaf experts in network'}
   )
   self._n_leaf = n_leaf
   self._n_sample = n_sample
   parent._init(config)
end

function Equanimizer:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample())
   
   local ostates = model:forward{
      input=batch:inputs(), carry={batch_indices=batch_indices},
      global={focus='experts'}
   }
   
   local loss, outputs = self._criterion:forward(
      ostates, batch:targets(), batch_indices
   )
   
   batch:setLoss(loss)  
   batch:setOutputs(outputs)
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneFeedback", 
                          report, batch)
   
   --[[ backpropagate ]]--
   local istates, cstates = self._criterion:backward(
      ostates, batch:targets(), batch:indices()
   )
   model:backward{
      output=istates, carry=cstates, global={focus='examples'}
   }
   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          report, batch)
end

------------------------------------------------------------------------
--[[ Shampoo ]]--
------------------------------------------------------------------------
local Shampoo, parent = torch.class("dp.Shampoo", "dp.Evaluator")

function Shampoo:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample())
   
   local ostates = model:evaluate{
      input=batch:inputs(), carry={batch_indices=batch_indices}
   }
   
   local loss, outputs = self._criterion:evaluate(
      ostates, batch:targets(), batch_indices
   )
   
   batch:setLoss(loss)  
   batch:setOutputs(outputs)
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(
      self:id():name() .. ':' .. "doneFeedback", report, batch
   )
   model:doneBatch()
end

function Shampoo:report()
   local report = parent.report(self)
   report.essrl = self._criterion:report()
   return report
end

function Shampoo:resetLoss()
   parent.resetLoss(self)
   self._criterion:resetStatistics()
end
