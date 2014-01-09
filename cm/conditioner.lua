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
      output=istates, carry=cstates, 
      global={focus='examples', scale=1/batch:nSample()}
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
-- Adds a second training phase to Conditioner.
-- It focuses on experts in order to impose an equanimous 
-- constraint that balances the distribution of expert-examples.
------------------------------------------------------------------------
local Equanimizer, parent = torch.class("dp.Equanimizer", "dp.Conditioner")

--[[function Equanimizer:__init(config)
   config = config or {}
   local args, xlua.unpack(
      {config},
      'Equanimizer', 
      'Adds a second training phase to Conditioner. '..
      'It focuses on experts in order to impose an equanimous '..
      'constraint that balances the distribution of expert-examples.',
   )
   parent._init(config)
end--]]

function Equanimizer:propagateBatch(batch, report) 
   -- focus on examples
   parent.propagateBatch(self, batch, report)
   
   -- focus on experts
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local batch_indices = torch.range(1,batch:nSample()):long()
   
   local ostates = model:forward{
      input=batch:inputs(), carry={batch_indices=batch_indices},
      global={focus='experts'}
   }
      
   --[[ backpropagate ]]--
   local istates, cstates = self._criterion:expertFocus(
      ostates, batch:targets(), batch:indices()
   )
   model:backward{
      output=istates, carry=cstates, 
      global={focus='examples', scale=1/batch:nSample()}
   }
   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
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
