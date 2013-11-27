----------------------------------------------------------------------
-- A Confusion Matrix class
--
-- Example:
-- conf = optim.ConfusionMatrix( {'cat','dog','person'} )   -- new matrix
-- conf:zero()                                              -- reset matrix
-- for i = 1,N do
--    conf:add( neuralnet:forward(sample), label )          -- accumulate errors
-- end
-- print(conf)                                              -- print matrix
-- image.display(conf:render())                             -- render matrix
--
local ConfusionMatrix = torch.class('optim.ConfusionMatrix2', 'optim.ConfusionMatrix')

function ConfusionMatrix:batchAdd(predictions, targets)
   local preds, targs, _
   if predictions:dim() == 1 then
      -- predictions is a vector of classes
      preds = predictions
   elseif predictions:dim() == 2 then
      -- prediction is a matrix of class likelihoods
      if predictions:size(2) == 1 then
         -- or prediction just needs flattening
         preds = predictions:copy()
      else
         _,preds = predictions:max(2)
      end
      preds:resize(preds:size(1))
   else
      error("predictions has invalid number of dimensions")
   end
      
   if targets:dim() == 1 then
      -- targets is a vector of classes
      targs = targets
   elseif targets:dim() == 2 then
      -- targets is a matrix of one-hot rows
      if targets:size(2) == 1 then
         -- or targets just needs flattening
         targs = targets:copy()
      else
         _,targs = targets:max(2)
      end
      targs:resize(targs:size(1))
   else
      error("targets has invalid number of dimensions")
   end
   --loop over each pair of indices
   for i = 1,preds:size(1) do
      self.mat[targs[i]][preds[i]] = self.mat[targs[i]][preds[i]] + 1
   end
end

