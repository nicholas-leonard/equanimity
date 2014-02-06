------------------------------------------------------------------------
--[[ EnFactory ]]--
-- An experiment builder using Enhanced models.
------------------------------------------------------------------------
local EnFactory, parent = torch.class("dp.EnFactory", "dp.MLPFactory")
EnFactory.isEnFactory = true
   
function EnFactory:__init(config)
   config = config or {}
   local args, name = xlua.unpack(
      {config},
      'EnFactory', nil,
      {arg='name', type='string', default='En'}
   )
   config.name = name
   parent.__init(self, config)
end

function EnFactory:buildModel(opt)
   local function addHidden(mlp, encoding, activation, input_size, layer_index)
      layer_index = layer_index or 1
      local output_size = math.ceil(
         opt.model_width * opt.layer_scales[layer_index]
      )
      local hidden_size = math.ceil(
         opt.model_width * opt.encoder_scales[layer_index]
      )
      mlp:add(
         dp.Enhancer{
            input_size=input_size, hidden_size=hidden_size, 
            output_size=output_size, eval_proto=opt.eval_proto,
            hidden_transfer=self:buildTransfer(encoding),
            output_transfer=self:buildTransfer(activation), 
            lambda=opt.mixture_coeffs[layer_index],
            update_scale=opt.update_scales[layer_index],
            bpae_coeff=opt.bpae_coeffs[layer_index],
            input_noise=self:buildDropout(opt.input_noises[layer_index])
         }
      )
      print(output_size .. " hidden neurons")
      if layer_index < (opt.model_dept-1) then
         return addHidden(mlp, encoding, activation, output_size, layer_index+1)
      else
         return output_size
      end
   end
   --[[Model]]--
   local mlp = dp.Sequential()
   -- hidden layer(s)
   print(opt.feature_size .. " input neurons")
   local last_size = addHidden(mlp, opt.encoding, opt.activation, opt.feature_size, 1)
   -- output layer
   mlp:add(
      dp.Neural{
         input_size=last_size, output_size=#opt.classes,
         transfer=nn.LogSoftMax()
      }
   )
   print(#opt.classes.." output neurons")
   --[[GPU or CPU]]--
   if opt.model_type == 'cuda' then
      require 'cutorch'
      require 'cunn'
      mlp:cuda()
   elseif opt.model_type == 'double' then
      mlp:double()
   elseif opt.model_type == 'float' then
      mlp:float()
   end
   return mlp
end


------------------------------------------------------------------------
--[[ PGNDTFactory ]]--
-- PostgreSQL Neural Decision Tree builder
------------------------------------------------------------------------
local PGEnFactory, parent = torch.class("dp.PGEnFactory", "dp.EnFactory")
PGEnFactory.isPGEnFactory = true
   
function PGEnFactory:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGEnFactory', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGEnFactory:buildObserver(opt)
   return {
      self._logger,
      dp.PGEarlyStopper{
         start_epoch = 11,
         pg = self._pg,
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.max_tries,
         save_strategy = self._save_strategy,
         min_epoch = 10, max_error = opt.max_error
      },
      dp.PGDone{pg=self._pg}
   }
end
