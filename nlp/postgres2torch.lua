require 'dp'
------------------------------------------------------------------------
-- Generate a serialized billion-words dataset from its 
-- PostgreSQL representation. Each word is an int. 
------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate torch dump for billion-words dataset')
cmd:text('Example:')
cmd:text('$> th postgres2torch.lua --dataset train')
cmd:text('Options:')
cmd:option('--dataset', 'train', 'train | valid | test')
cmd:option('--stepSize', 1000, 'amount of sentences to retrieve per query')
cmd:option('--wordMap', false, 'outputs a mapping of word strings to word integers')
opt = cmd:parse(arg or {})

local pg = dp.Postgres()
local data_path = paths.concat(dp.DATA_DIR, 'billion-words')
check_and_mkdir(data_path)

if opt.wordMap then
   local word_map = {}
   local rows = pg:fetch(
      "SELECT word_id, word_str, word_count " ..
      "FROM bw.word_count " ..
      "ORDER BY word_count DESC"
   )
   local word_freq = torch.IntTensor(table.length(rows)):zero()
   for j, row in ipairs(rows) do
      local word_id = tonumber(row[1])
      local word_str = row[2]
      local word_count = tonumber(row[3])
      word_freq[word_id] = word_count
      word_map[word_id] = word_str
   end
   torch.save(paths.concat(data_path, 'word_freq.th7'), word_freq)
   torch.save(paths.concat(data_path, 'word_map.th7'), word_map)
   os.exit()
end

local n_word = tonumber(
   pg:fetchOne(
      "SELECT SUM(array_upper(sentence_words, 1)-1) FROM bw.%s_sentence", 
      {opt.dataset}
   )[1]
)

local data = torch.IntTensor(n_word, 2)
--sequence of words where sentences are delimited by </s>
local corpus = data:select(2, 2)
--holds start indices of sentence of word at same index in corpus
local delimiters = data:select(2, 1)

-- contenxt_size of n-gram (i.e.: n-1)
local context_size = opt.contextSize

local n_sentence = tonumber(
   pg:fetchOne(
      "SELECT MAX(sentence_id) FROM bw.%s_sentence ", 
      {opt.dataset}
   )[1]
)
local step = opt.stepSize
local start_time = os.time()
print("loading " .. n_word .. " words or " .. n_sentence .. " sentences from db")

local sentence_idx = 1
for i=1,n_sentence,step do
   local rows = pg:fetch(
      "SELECT sentence_id, sentence_words " ..
      "FROM bw.%s_sentence " ..
      "WHERE sentence_id BETWEEN %d and %d", 
      {opt.dataset, i, math.min(i+step-1, n_sentence)}
   )
   for j, row in ipairs(rows) do
      local sentence_id = tonumber(row[1])
      local sentence_words = torch.IntTensor(
         _.map(
            _.split(string.sub(row[2], 9, -2), ','), 
            function(str) return tonumber(str) end
         )
      )
      local sentence_size = sentence_words:size(1)
      corpus:narrow(1, sentence_idx, sentence_size):copy(sentence_words)
      delimiters:narrow(1, sentence_idx, sentence_size):fill(sentence_idx)
      sentence_idx = sentence_idx + sentence_size
   end
   xlua.progress(i, n_sentence)
end
print("\nloaded " .. n_sentence .. " sentences in " .. os.time()-start_time .. " sec.")

--print(ngrams[{{ngrams:size(1)-100, ngrams:size(1)},{}}])
--print(corpus[{{corpus:size(1)-100, corpus:size(1)}}])


torch.save(paths.concat(data_path, opt.dataset..'_data.th7'), data)
