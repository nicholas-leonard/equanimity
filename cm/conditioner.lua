------------------------------------------------------------------------
--[[ Conditioner ]]--
------------------------------------------------------------------------
local Conditioner, parent = torch.class("dp.Conditioner", "dp.Optimizer")

function Conditioner:__init()
   
end
      
function Conditioner:propagateBatch(batch)   
   local model = self._model
   --[[ Phase 1 : Focus on examples ]]--
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local ostate = model:forward{
      input={
         act = batch:inputs(),
         indices = batch:indices()
      },
      focus='examples'
   }
   
   local loss, outputs = 
      self._criterion:forward(ostate, batch:targets(), batch:indices())
   batch:setLoss(loss)  
   batch:setOutputs(outputs)
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneFeedback", 
                          self:report(), batch)
   
   print("END TEST")
   os.exit()
   --[[ backpropagate ]]--
   self._criterion:backward(ostate, batch:targets(), batch:indices())
   model:backward{output=batch:outputGradients()}

   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          self:report(), batch)
                          
   --[[ Phase 2 : Focus on experts ]]--
   -- sample a batch of experts for phase 2
   local experts = self._expert_sampler:sampleBatch()
   gstate = {focus='experts'}
end
