require 'xlua'
require 'sys'
require 'torch'

local Learner = torch.class("optim.Learner")

function Learner:__init(...)   
   args, self.dataset, self.batch_size, self.type, self.logger, self.plot
      = xlua.unpack(
      {...},
      'Learner constructor', nil,
      {arg='dataset', type='table', help='an object that implements the Dataset interface', req=true}
      {arg='batch_size', type='number', help='number of examples per mini-batch', default=32},
      {arg='type', type='string', help='type: double | float | cuda', default='double'},
      {arg='logger', type='table', help='an object implementing the Logger interface', default=TODO},
      {arg='plot', type='boolean', help='live plot of confusion matrix', default=false},
   )
   self.epoch = 1
   self.shuffle = self.shuffle or false
end

function Learner:doEpoch()
   -- local vars
   local time = sys.clock()

   local nextBatch
   if self.shuffle then
      -- shuffle at each epoch
      shuffle = torch.randperm(self.dataset:size())
      function nextBatch(start, stop)
         local indices = shuffle:sub(start,stop):long()
         local inputs = self.dataset:inputs():index(1, indices)
         local targets = self.dataset:targets():index(1, indices)
         return inputs, targets
      end
   else
      function nextBatch(start, stop)
         local inputs = self.dataset:inputs():sub(start, stop)
         local targets = self.datasets:targets():sub(start, stop)
      end
   end

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. self.batch_size .. ']')
   for start = 1,self.dataset:size(),self.batch_size do
      local stop = math.min(start+self.batch_size-1,self.dataset:size())
      -- disp progress
      xlua.progress(start, self.dataset:size())

      -- create mini batch
      self:doBatch(nextBatch(start, stop))
   
   -- time taken
   time = sys.clock() - time
   time = time / self.dataset:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   
   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end      

      
function Learner:doBatch(inputs, targets)

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
        -- get new parameters
        if x ~= parameters then
           parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        --[[feedforward]]--
        -- evaluate function for complete mini batch
        local outputs = model:forward(inputs)
        -- average loss (a scalar)
        local f = criterion:forward(outputs, targets)
        
        --[[backpropagate]]--
        -- estimate df/do (o is for outputs), a tensor
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
         
        --[[measure error]]--
        -- update confusion
        confusion:batchAdd(outputs, targets)
                       
        -- return f and df/dX
        return f,gradParameters
     end

   optim.sgd(feval, parameters, optimState)
end


local Trainer = torch.class("optim.Trainer", "optim.Learner")

function Trainer:__init(...)
   args, self.learning_rate, self.weight_decay, self.momentum, self.shuffle
      = xlua.unpack(
      {...},
      'Trainer constructor', nil,
      {arg='learning_rate', type='number', help='learning rate at start of learning', req=true},
      {arg='weight_decay', type='number', help='weight decay coefficient', default=0},
      {arg='momentum', type='number', help='momentum of the parameter gradients', default=0},
      {arg='shuffle', type='boolean', help='shuffles the dataset after each epoch', default=true
   )
   parent.__init(self, args)
end

function Trainer:doEpoch(...)

end


local Tester = torch.class("optim.Tester", "optim.Tester")

function Tester:__init(...)

end

function Tester:doEpoch(...)

end
