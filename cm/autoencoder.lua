------------------------------------------------------------------------
--[[ AntoEncoder ]]--
-- An [online] [denoising] autoencoder
------------------------------------------------------------------------
local AutoEncoder, parent = torch.class("dp.AutoEncoder", "dp.Container")
AutoEncoder.isAutoEncoder = true

function AutoEncoder:__init(config)
   config = config or {}
   local args, input_size, hidden_size, hidden_transfer, 
         output_transfer, input_noise, tied_weights, typename, 
         sparse_init, criterion = xlua.unpack(
      {config},
      'AutoEncoder', 
      'An [online] [denoising] autoencoder',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hidden_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='hidden_transfer', type='nn.Module', default=nn.Tanh(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='output_transfer', type='nn.Module', default=nn.Tanh(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='input_noise', type='nn.Dropout', 
       help='applies dropout to the inputs of this model, i.e. '..
       'corrupts the input as in a denoising auto-encoder (DAE)'},
      {arg='tied_weights', type='boolean', default=true,
       help='ties weights of encoder and decoder'},
      {arg='typename', type='string', default='autoencoder', 
       help='identifies Model type in reports.'},
      {arg='sparse_init', type='boolean', default=true},
      {arg='criterion', type='nn.Criterion', default=nn.MSECriterion(),
       help='criterion used to evaluate reconstruction error'}
   )
   config.typename = typename
   self._encoder = dp.Neural{
      input_size=input_size, output_size=hidden_size,
      transfer=hidden_transfer, dropout=input_noise,
      sparse_init=sparse_init
   }
   self._decoder = dp.Neural{
      input_size=hidden_size, output_size=input_size,
      transfer=output_transfer, sparse_init=sparse_init
   }
   self._tied_weights = tied_weights
   if tied_weights then
      self._decoder:parameters().weight.param:set(
         self._encoder:parameters().weight.param:t()
      )
   end
   self._criterion = criterion
   parent.__init(self, config)
   self._models = {self._encoder, self._decoder}
   self:zeroStatistics()
end

function AutoEncoder:setup(config)
   parent.setup(self, config)
   local predecessor = self._predecessor
   config.container = self
   local names = {'encoder', 'decoder'}
   for i, model in ipairs(self._models) do
      config.id = self:id():create(names[i])
      config.predecessor = predecessor
      predecessor = model
      config.successor = self._models[i+1] or self._successor
      model:setup(config)
   end
   self._data_view = self._models[1]:dataView()
end

function AutoEncoder:_forward(cstate)
   -- encoder state
   self._estate, cstate = self._encoder:forward{
      input=self.istate.act, carry=cstate, global=self.gstate
   }
   self.ostate = self._estate
   -- decoder state
   self._dstate = self._decoder:forward{
      input=self._estate, carry=cstate, global=self.gstate
   }
   --- criterion : learn to recontruct input activation 
   self._d_act = self._decoder.ostate.act:double()
   self._target = self.istate.act:double()
   local loss = self._criterion:forward(self._d_act, self._target)
   self._loss = self._loss + self.istate.act:size(1) * loss
   self._sample_count = self._sample_count + self.istate.act:size(1)
   return cstate
end

function AutoEncoder:_backward(cstate)
   -- criterion
   self._dstate.grad = self._criterion:backward(
      self._d_act, self._target
   ):type(self.istate.act:type())
   -- decoder
   self._estate, cstate = self._decoder:backward{
      output=self._dstate, carry=cstate, global=self.gstate
   }
   self.ostate = self._estate
   -- encoder
   self.istate, cstate = self._encoder:backward{
      output=self._estate, carry=cstate, global=self.gstate
   }
   return cstate
end

function AutoEncoder:report()
   local report = parent.report(self) or {}
   report.loss = self._loss / self._sample_count
   print(self:name(), report.loss)
   return report
end

function AutoEncoder:zeroStatistics()
   parent.zeroStatistics(self)
   self._loss = 0
   self._sample_count = 0
end
