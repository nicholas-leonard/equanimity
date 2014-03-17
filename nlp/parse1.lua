require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('LM parser')
cmd:text('Example:')
cmd:text('$> th parse1.lua --path /home/nicholas/corpus /home/nicholas/corpus2')
cmd:text('Options:')
cmd:option('--path', '/home/nicholas/corpus', 'path to corpus')

pg = dp.Postgres()
opt = cmd:parse(arg or {})

function parse()
   local wc = {}
   local start = os.time()
   local i = 0
   for __, path in ipairs(_.split(opt.path, ' ')) do
      for file in lfs.dir(path) do
         if #file > 2 then 
            local filename = paths.concat(path, file)
            print(i, filename)
            for line in io.lines(filename) do
               for token_i, token_str in ipairs(_.split(line, ' ')) do
                  wc[token_str] = (wc[token_str] or 0) + 1
               end
            end
            i = i + 1
         end
      end
   end
   local unknown_count = 0
   start = os.time()
   i = 0
   local j = 0
   for word, count in pairs(wc) do
      if count < 3 then
         unknown_count = unknown_count + count
      else
         print(i, word, count)
         -- http://stackoverflow.com/questions/12316953/insert-varchar-with-single-quotes-in-postgresql
         pg:execute(
            "INSERT INTO bw.word_count (word_str, word_count) " ..
            "VALUES ($token093$%s$token093$, %d)", {word, count} 
         )
         i = i + 1
      end
   end
   pg:execute(
      "INSERT INTO bw.word_count (word_str, word_count) " ..
      "VALUES ('<UNK>', %d)", {unknown_count} 
   )
   print("persisted " .. i .. " words in " .. os.time()-start .. " sec.")
   print("done")
end

parse()
