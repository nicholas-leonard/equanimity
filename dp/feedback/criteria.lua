------------------------------------------------------------------------
--[[ Criteria ]]--
-- Feedback
-- Adapter that feeds back and accumulates the error of one or many
-- nn.Criterion. Each supplied nn.Criterion requires a name for 
-- reporting purposes. Default name is typename minus module name(s)
------------------------------------------------------------------------
local Criteria, parent 
   = torch.class("dp.Criteria", "dp.Feedback")


function Criteria:__init(config)
   local args, criteria, name, typename_pattern
      = xlua.unpack(
      {config},
      'Criteria', nil,
      {arg='criteria', type='nn.Criterion | table', req=true,
       help='list of criteria to monitor'},
      {arg='name', type='string', default='criteria'},
      {arg='typename_pattern', type='string', 
       help='require criteria to have a torch.typename that ' ..
       'matches this pattern', default="^nn[.]%a*Criterion$"}
   )
   config.name = name
   parent.__init(self, config)
   
   self._criteria = {}
   self._name = name
   if torch.typename(criteria) then
      criteria = {criteria}
   end
   
   for k,v in pairs(criteria) do
      -- non-list items only
      if type(k) ~= 'number' then
         self._criteria[k] = v
      end
   end
   
   for i,v in ipairs(criteria) do
      -- for listed criteria, default name is derived from typename
      local k = _.split(torch.typename(criteria), '[.]')
      k = k[#k]
      -- remove suffix 'Criterion'
      if string.sub(k, -9) == 'Criterion' then
         k = string.sub(k, 1, -10)
      end
      -- make lowercase
      k = string.lower(k)
      self._criteria[k] = v
   end
   if typepattern ~= '' then
      for k,v in pairs(self._criteria) do
         assert(typepattern(v,typepattern), "Invalid criteria typename")
      end
   end
   
   self._errors = {}
   self:reset()
end

function Criteria:reset()
   -- reset error sums to zero
   for k,v in self._criteria do
      self._errors[k] = 0
   end
   self._samples_seen = 0
end

function Criteria:add(batch)             
   local current_error
   for k,v in self._criteria do
      current_error = v:forward(batch:outputs(), batch:targets())
      self._errors[k] =  (
                              ( self._samples_seen * self._errors[k] ) 
                              + 
                              ( batch:nSample() * current_error )
                         ) 
                         / 
                         self._samples_seen + batch:nSample()
      --TODO gather statistics on backward outputGradients?
   end
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Criteria:report()
   return { 
      [self:name()] = self._errors,
      n_sample = self._samples_seen
   }
end
