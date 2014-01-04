require 'cm'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Plot Channel Curves')
cmd:text('Example:')
cmd:text('$> th plotcurves.lua --entry 1637 --channel "optimizer:feedback:confusion:accuracy,validator:feedback:confusion:accuracy,tester:feedback:confusion:accuracy" --curveName "train,valid,test" -x "epoch", -y "accuracy"')
cmd:text('Options:')
cmd:option('--entry', -1, 'xplog entry number')
cmd:option('--channel', 'optimizer:feedback:confusion:accuracy,validator:feedback:confusion:accuracy,tester:feedback:confusion:accuracy', 'comma-seperated channel names')
cmd:option('--name', 'train,valid,test', 'comma-seperated curve names')
cmd:option('-x', 'epoch', 'name of the x-axis')
cmd:option('-y', 'accuracy', 'name of the y-axis')
cmd:text()
opt = cmd:parse(arg or {})

local entry = dp.PGXpLogEntry{id=opt.entry,pg=dp.Postgres()}
local hp = entry:hyperReport().hyperparam
print(hp)
entry:plotReportChannel{
   channels=opt.channel, curve_names=opt.name, 
   x_name=opt.x, y_name=opt.y
}
