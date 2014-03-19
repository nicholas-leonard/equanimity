require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('LM parser')
cmd:text('Example:')
cmd:text('$> th parse1.lua --path /home/nicholas/corpus /home/nicholas/corpus2')
cmd:text('Options:')
cmd:option('--path', '/home/nicholas/corpus', 'path to corpus')
cmd:option('--set', 'train', 'train | valid | test')
cmd:option('--transactionSize', 100, 'number of queries per transaction')

pg = dp.Postgres()
opt = cmd:parse(arg or {})

function parse()
   local rows = pg:fetch("SELECT word_str, word_id FROM bw.word_count")
   local voc = {}
   print("vocabulary has " .. #rows .. " words")
   for i, row in ipairs(rows) do 
      voc[row[1]] = row[2]
   end
   local unk_id, start_id, end_id = voc['<UNK>'], voc['<S>'], voc['</S>']
   assert((unk_id ~= nil) and (start_id ~= nil) and (end_id ~= nil))
   local queries = {}
   local start = os.time()
   line_i = 0
   for file in lfs.dir(opt.path) do
      if #file > 2 then 
         local filename = paths.concat(opt.path, file)
         print(line_i, filename)
         for line in io.lines(filename) do
            local sentence = {start_id}
            for token_i, token_str in ipairs(_.split(line, ' ')) do
               table.insert(sentence, voc[token_str] or unk_id)
            end
            table.insert(sentence, end_id)
            line_i = line_i + 1
            local query = string.format(
               "INSERT INTO bw." .. opt.set .. "_sentence " ..
               "(sentence_id, sentence_words) " ..
               "(SELECT %d, ARRAY[%s]::INT4[]);", 
               line_i, _.join(sentence, ", ")
            )
            table.insert(queries, query)
            if line_i % opt.transactionSize == 0 then
               pg:execute(_.join(queries, ' '))
               queries = {}
            end
         end
      end
   end
   print(#queries)
   if #queries > 0 then
      pg:execute(_.join(queries, ' '))
   end
   print("parsed " .. line_i .. " lines in " .. os.time()-start .. "sec.")
   print"done"
end

parse()
