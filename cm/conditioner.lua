------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Optimizer")

function Conditioner:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample()):long()
   
   local ostates = model:forward{
      input=batch:inputs(), carry={batch_indices=batch_indices}
   }
   
   local loss, outputs, istates, cstates = self._criterion:forward(
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

   model:backward{
      output=istates, carry=cstates, global={scale=1/batch:nSample()}
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
--[[ Shampoo ]]--
------------------------------------------------------------------------
local Shampoo, parent = torch.class("dp.Shampoo", "dp.Evaluator")

function Shampoo:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample()):long()
   
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
