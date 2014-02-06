------------------------------------------------------------------------
--[[ Enhancer ]]--
-- experimental (probably won't work)
-- A neural layer supported by an auto-encoder
-- Contingencies : LE gets more than one update...
------------------------------------------------------------------------
local Enhancer, parent = torch.class("dp.Enhancer", "dp.Container")
Enhancer.isEnhancer = true

function Enhancer:__init(config)
   config = config or {}
   local args, input_size, hidden_size, output_size, hidden_transfer, 
         output_transfer, input_noise, typename, sparse_init, 
         criterion, update_scale, eval_proto, lambda, bpae_coeff
      = xlua.unpack(
      {config},
      'Neural', 
      'An [online] [denoising] autoencoder',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hidden_size', type='number', req=true,
       help='Number of hidden neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='hidden_transfer', type='nn.Module', default=nn.Sigmoid(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='output_transfer', type='nn.Module', default=nn.Tanh(),
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='input_noise', type='nn.Dropout', 
       help='applies dropout to the inputs of the layer-encoder, '..
       'corrupts the input as in a denoising auto-encoder'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'},
      {arg='sparse_init', type='boolean', default=true},
      {arg='criterion', type='nn.Criterion', default=nn.MSECriterion(),
       help='criterion used to evaluate reconstruction error'},
      {arg='update_scale', type='number', default=0.1,
       help='learning rate for substracting act gradient from act'},
      {arg='eval_proto', type='string', default='layer-only',
       help='evaluation protocol: layer-only | average | product'},
      {arg='lambda', type='number', default=0.5,
       help='mixture coefficient. Weight of layer activations'},
      {arg='bpae_coeff', type='number', default=0,
       help='backpropagate auto-encoder coeff*gradients into layer'}
   )
   self._encoder = dp.Neural{
      input_size=output_size, output_size=hidden_size,
      transfer=hidden_transfer, dropout=input_noise,
      sparse_init=sparse_init
   }
   self._decoder = dp.Neural{
      input_size=hidden_size, output_size=output_size,
      transfer=output_transfer, sparse_init=sparse_init
   }
   self._layer = dp.Neural{
      input_size=input_size, output_size=output_size,
      transfer=output_transfer:clone(), sparse_init=sparse_init
   }
   self._criterion = criterion
   self._update_scale = update_scale
   self._eval_proto = eval_proto
   self._lambda = lambda
   self._bpae_coeff = bpae_coeff
   config.typename = typename or 'enhancer'
   parent.__init(self, config)
   self._models = {self._encoder, self._decoder, self._layer}
   self:zeroStatistics()
end

function Enhancer:setup(config)
   parent.setup(self, config)
   local predecessor = self._predecessor
   config.container = self
   local names = {'encoder', 'decoder', 'layer'}
   local successors = {self._decoder}
   local predecessors = {[2] = self._decoder}
   for i, model in ipairs(self._models) do
      config.id = self:id():create(names[i])
      config.predecessor = predecessors[i] or self._predecessor
      config.successor = successors[i] or self._successor
      model:setup(config)
   end
   self._data_view = self._models[3]:dataView()
end


function Enhancer:_forward(cstate)
   -- layer
   self.ostate, cstate = self._layer:forward{
      input=self.istate, carry=cstate, global=self.gstate
   }
   -- encoder
   local ostate, ecstate = self._encoder:forward{
      input=self.ostate.act, carry=cstate, global=self.gstate
   }
   -- decoder
   self._decoder:forward{
      input=ostate, carry=ecstate, global=self.gstate
   }
   return cstate
end

function Enhancer:_backward(cstate)
   -- auto-encoder 
   --- criterion : learn to predict desired activation 
   local d_act = self._decoder.ostate.act:double()
   local target = self.ostate.act:clone():add(-self._update_scale, self.ostate.grad):double()
   local loss = self._criterion:forward(d_act, target)
   self._loss = self._loss + self.istate.act:size(1) * loss
   self._sample_count = self._sample_count + self.istate.act:size(1)
   local d_grad = self._criterion:backward(d_act, target):type(self.istate.act:type())
   --- decoder
   local istate, cstate = self._decoder:backward{
      output=d_grad, carry=cstate, global=self.gstate
   }
   --- encoder
   local istate = self._encoder:backward{
      output=istate, carry=cstate, global=self.gstate
   }
   -- merge auto-encoder grad with layer grad
   if self._bpae_coeff > 0 then
      self.ostate.grad:add(self._bpae_coeff, istate.grad)
   end
   -- layer
   self.istate, cstate = self._layer:backward{
      output=self.ostate, carry=cstate, global=self.gstate
   }
   return cstate
end

function Enhancer:_evaluate(cstate)
   -- layer
   self.ostate, cstate = self._layer:evaluate{
      input=self.istate, carry=cstate, global=self.gstate
   }
   -- merge activations during evaluation (mixture model / smoothing)
   if self._eval_proto ~= 'layer-only' then
      -- encoder
      local __, ecstate = self._encoder:evaluate{
         input=self.ostate.act:clone(), carry=cstate, global=self.gstate
      }
      -- decoder
      self._decoder:evaluate{
         input=self._encoder.ostate, carry=ecstate, global=self.gstate
      }
      if self._eval_proto == 'average' then
         self.ostate.act:mul(self._lambda):add(1-self._lambda, self._decoder.ostate.act):div(2)
      elseif self._eval_proto == 'product' then
         self.ostate.act:pow(self._lambda):cmul(self._decoder.ostate.act:pow(1-self._lambda))
      else
         error"unknown eval protocol"
      end
   end
   return cstate
end

function Enhancer:report()
   local report = parent.report(self) or {}
   report.loss = self._loss / self._sample_count
   print(self:name(), report.loss)
   return report
end

function Enhancer:zeroStatistics()
   parent.zeroStatistics(self)
   self._loss = 0
   self._sample_count = 0
end
