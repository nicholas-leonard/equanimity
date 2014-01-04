require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('List experiments by descending minima')
cmd:text('Example:')
cmd:text('$> th plotcurves.lua --entry 1637 --channel "optimizer:feedback:confusion:accuracy,validator:feedback:confusion:accuracy,tester:feedback:confusion:accuracy" --curveName "train,valid,test" -x "epoch", -y "accuracy"')
cmd:text('Options:')
cmd:option('--collection', '', 'comma separated xplog collection id')
cmd:option('--limit', 10, 'number of top experiments to list')
cmd:text()
opt = cmd:parse(arg or {})


local xplog = dp.PGXpLog()
local rows = xplog:listMinima(opt.collection, opt.limit)
local entries = {}
local head = {minima={},epoch={},xp_id={}}
for i, row in ipairs(rows) do
   head.minima[i] = tonumber(string.sub(row.minima,1,7))
   head.epoch[i] = tonumber(row.epoch)
   head.xp_id[i] = tonumber(row.xp_id)
   table.insert(entries, xplog:entry(row.xp_id))
end

local sheet = {}
for i, entry in ipairs(entries) do
   for k,v in pairs(entry:hyperReport().hyperparam) do
      local values = sheet[k] or {}
      if type(v) == 'table' then
         v = table.tostring(v)
      end
      values[i] = v
      sheet[k] = values
   end
end

print"------------RANK BY MINIMA----------------"
for k,v in pairs(head) do
   print(k, table.tostring(v))
end
for i,k in ipairs(_.sort(_.keys(sheet))) do
   if k ~= 'classes' then
      print(k, table.tostring(sheet[k]))
   end
end



