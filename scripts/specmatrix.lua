require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('View experiment specialization matrices at minima')
cmd:text('Example:')
cmd:text('$> th specmatrix.lua --entry 1654 ')
cmd:text('Options:')
cmd:option('--entry', '', 'experiment id')
cmd:option('--hp', false, 'show hypeparams')
cmd:option('--epoch', -1, 'epoch of spec matrix. defaults to minima')
cmd:option('--window', 20, 'nb of epochs after epoch')
cmd:option('--popularity', false, 'plot popularity through time')
cmd:option('--tester', false, 'plot test spec instead of optimizer')
cmd:text()
opt = cmd:parse(arg or {})

local entry = dp.PGXpLogEntry{id=opt.entry,pg=dp.Postgres()}
if opt.hp then
   local hp = entry:hyperReport().hyperparam
   print(hp)
end

local report
if opt.epoch < 0 then
   local epochs, minimas, epoch, minima = entry:minima() 
   report = entry:report(epoch)
else
   report = entry:report(opt.epoch)
end
local channel = {'optimizer','essrl','spec'}
if opt.tester then
   channel = {'tester','essrl','spec'}
end
print"spec matrix"
opt_sm = table.channelValue(report, channel)
print(opt_sm:long())
require 'gnuplot'
local pop
if opt.popularity then
   pop = torch.Tensor(opt_sm:size(1), opt.window):zero()
   for i=1,opt.window do
      pop[{{},i}] = opt_sm:sum(2) --opt_sm:max(2):cdiv(opt_sm:sum(2))
      report = entry:report(opt.epoch+i)
      opt_sm = table.channelValue(report, channel)
   end
   gnuplot.imagesc(pop,'color')
   --gnuplot.splot(pop)
else
   gnuplot.imagesc(opt_sm,'color')
end
--[[
local pop2
opt_sm = table.channelValue(report, {'tester','essrl','spec'})
if opt.popularity then
   pop2 = torch.Tensor(opt_sm:size(1), opt.window):zero()
   for i=1,opt.window do
      pop2[{{},i}] = opt_sm:sum(2)
      report = entry:report(opt.epoch+i)
      opt_sm = table.channelValue(report, {'tester','essrl','spec'})
   end
   --gnuplot.imagesc(pop,'color')
   --gnuplot.splot(pop)
end
gnuplot.imagesc(torch.add(pop,-pop2),'color')--]]
