require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('LM parser')
cmd:text('Example:')
cmd:text('$> th parser.lua --path /home/nicholas/corpus/')
cmd:text('Options:')
cmd:option('--path', '/home/nicholas/corpus', 'path to corpus')

pg = Postgres()
opt = cmd:parse(arg or {})

function parse1()
   local wc = {}
   for file in lfs.dir(opt.path) do
      if #file > 2 then 
         local filename = paths.concat(opt.path, file)
         for line in io.lines(filename) do
            for token_i, token_str in ipairs(_.split(line)) do
               wc[token_str] = (wc[token_str] or 0) + 1
            end
         end
      end
   end
   return wc
end

function parse2()
   line_i = 1
   for line in io.lines(filename) do
      for token_i, token_str in ipairs(_.split(line)) do
         print"None"
      end
      line_i = line_i + 1
   end
end

function parseAll()
   local wc = parse1()
   local unknowns = {}
   local unknown_count = 0
   for word, count in pairs(wc) do
      if count < 3 then
         unknowns[word] = true
         unknown_count = unknown_count + count
      end
   end
   for word, count in pairs(unknown) do
      pg.execute(
         "INSERT INTO bw
   end
end
