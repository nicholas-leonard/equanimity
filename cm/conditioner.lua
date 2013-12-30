------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Optimizer")

function Conditioner:propagateBatch(batch, report)   
   local model = self._model
   --[[ Phase 1 : Focus on examples ]]--
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
                          
   --[[ Phase 2 : Focus on experts ]]--
   -- sample a batch of experts for phase 2
   --local experts = self._expert_sampler:sampleBatch()
   --gstate = {focus='experts'}
end

function Conditioner:report()
   local report = parent.report(self)
   report.essrl = self._criterion:report()
   print(report.essrl)
   return report
end

function Conditioner:resetLoss()
   parent.resetLoss(self)
   self._criterion:resetStatistics()
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
end