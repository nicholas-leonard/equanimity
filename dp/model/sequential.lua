------------------------------------------------------------------------
--[[ Sequential ]]--
-- Model, Adapter, Composite
-- Replaces nn.Sequential such that it can be used for both 
-- optimzation and evaluation.
------------------------------------------------------------------------

local Sequential, parent = torch.class("dp.Sequential", "dp.Container")

function Sequential:__init(config)
   config = config or {}
   config.typename = 'sequential'
   parent.__init(self, config)
end

function Sequential:extend(models)
   for model_idx, model in ipairs(models) do
      self:add(model)
   end
end

function Sequential:setup(config)
   parent.setup(self, config)
   local predecessor = self._predecessor
   config.container = self
   for i, model in ipairs(self._models) do
      config.id = self:id():create('s'..i)
      config.predecessor = predecessor
      predecessor = model
      config.successor = self._models[i+1] or self._successor
      model:setup(config)
   end
   self._data_view = self._models[1]:dataView()
end

function Sequential:report()
   local report = {typename=self._typename}
   for i, model in ipairs(self._models) do 
      report[i] = model:report()
   end
   return report
end

function Sequential:add(model)
   if #self._models == 0 then
      self.istate = model.istate
   end
   table.insert(self._models, model)
   self.ostate = model.ostate
   return self
end

function Sequential:size()
   return #self._models
end

function Sequential:get(index)
   return self._models[index]
end

function Sequential:_forward(gstate)
   local istate = self.istate
   for i=1,#self._models do 
      local model = self._models[i]
      istate = model:forward{input=istate,global=gstate}
   end
   self.ostate = istate
end

function Sequential:_backward(gstate, scale)
   scale = scale or 1
   local ostate = self.ostate
   for i=#self._models,1,-1 do
      ostate = self._models[i]:backward({output=ostate,global=gstate}, scale)
   end
   return ostate
end

function Sequential:_accept(visitor)
   for i=1,#self._models do 
      self._models[i]:accept(visitor)
   end 
   visitor:visitContainer(self)
end

function Sequential:zeroGradParameters()
  for i=1,#self._models do
     self._models[i]:zeroGradParameters()
  end
end

function Sequential:_update(gstate)
   for i=1,#self._models do
      self._models[i]:update(gstate)
   end
end

function Sequential:reset(stdv)
   for i=1,#self._models do
      self._models[i]:reset(stdv)
   end
end

function Sequential:parameters()
   error"NotImplementedError"
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   for i=1,#self._models do
      local params = self._models[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'dp.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self._models do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self._models do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self._models[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
