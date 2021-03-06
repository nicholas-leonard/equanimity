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
cmd:text('$> th postgres2torch.lua --treeTable "bw.word_cluster" --treeFile "word_tree1.th7"')
cmd:text('Options:')
cmd:option('--dataset', 'train', 'train | valid | test | small | tiny')
cmd:option('--stepSize', 1000, 'amount of sentences to retrieve per query')

cmd:option('--wordMap', false, 'outputs a mapping of word strings to word integers')

cmd:option('--treeTable', '', 'Name of the table containing a hierarchy of words. Used for hierarchical softmaxes.')
cmd:option('--treeFile', '', 'Name of the file where to save the torch dump.')

opt = cmd:parse(arg or {})

local pg = dp.Postgres()
local data_path = paths.concat(dp.DATA_DIR, 'BillionWords')
dp.check_and_mkdir(data_path)

if opt.treeTable ~= '' then
   assert(opt.treeFile ~= '', 'specify a file to save the torch dump')
   local rows = pg:fetch(
      "SELECT parent_id, child_ids FROM "..opt.treeTable
   )
   local tree = {}
   for j, row in ipairs(rows) do
      local parent_id, child_ids = row[1], row[2]
      local children = torch.IntTensor(table.fromString(child_ids))
      tree[tonumber(parent_id)] = children
   end
   torch.save(paths.concat(data_path, opt.treeFile), tree)
   os.exit()
end

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

if opt.dataset == 'small' then
   local step = opt.stepSize
   local n_word = tonumber(
      pg:fetchOne(
         "SELECT SUM(array_upper(sentence_words, 1)-1) "..
         "FROM bw.train_sentence AS a, bw.cluster5 AS b " ..
         "WHERE sentence_id = item_key"
      )[1]
   )
   local data = torch.IntTensor(n_word, 2)
   --sequence of words where sentences are delimited by </s>
   local corpus = data:select(2, 2)
   --holds start indices of sentence of word at same index in corpus
   local delimiters = data:select(2, 1)
   local n_cluster = tonumber(
      pg:fetchOne("SELECT MAX(cluster_key) FROM bw.cluster5")[1]
   )
   local sentence_idx = 1
   for i=1,n_cluster,step do
      local rows = pg:fetch(
         "SELECT sentence_id, sentence_words " ..
         "FROM bw.train_sentence AS a, bw.cluster5 AS b " ..
         "WHERE cluster_key BETWEEN %d and %d AND sentence_id = item_key", 
         {i, math.min(i+step-1, n_cluster)}
      )
      for j, row in ipairs(rows) do
         local sentence_id = tonumber(row[1])
         local sentence_words = torch.IntTensor(
            _.map(
               _.split(string.sub(row[2], 9, -2), ','), 
               function(k, str) return tonumber(str) end
            )
         )
         local sentence_size = sentence_words:size(1)
         corpus:narrow(1, sentence_idx, sentence_size):copy(sentence_words)
         delimiters:narrow(1, sentence_idx, sentence_size):fill(sentence_idx)
         sentence_idx = sentence_idx + sentence_size
      end
      xlua.progress(i, n_cluster)
   end
   torch.save(paths.concat(data_path, 'train_small.th7'), data)
   os.exit()
elseif opt.dataset == 'tiny' then
   local step = opt.stepSize
   local n_word = tonumber(
      pg:fetchOne(
         "SELECT SUM(array_upper(sentence_words, 1)-1) "..
         "FROM bw.train_sentence AS a " ..
         "WHERE sentence_id <= 38000"
      )[1]
   )
   local data = torch.IntTensor(n_word, 2)
   --sequence of words where sentences are delimited by </s>
   local corpus = data:select(2, 2)
   --holds start indices of sentence of word at same index in corpus
   local delimiters = data:select(2, 1)
   local n_sentence = 38000
   local sentence_idx = 1
   for i=1,n_sentence,step do
      local rows = pg:fetch(
         "SELECT sentence_id, sentence_words " ..
         "FROM bw.train_sentence " ..
         "WHERE sentence_id BETWEEN %d and %d", 
         {i, math.min(i+step-1, n_sentence)}
      )
      for j, row in ipairs(rows) do
         local sentence_id = tonumber(row[1])
         local sentence_words = torch.IntTensor(
            _.map(
               _.split(string.sub(row[2], 9, -2), ','), 
               function(k, str) return tonumber(str) end
            )
         )
         local sentence_size = sentence_words:size(1)
         corpus:narrow(1, sentence_idx, sentence_size):copy(sentence_words)
         delimiters:narrow(1, sentence_idx, sentence_size):fill(sentence_idx)
         sentence_idx = sentence_idx + sentence_size
      end
      xlua.progress(i, n_sentence)
   end
   torch.save(paths.concat(data_path, 'train_tiny.th7'), data)
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


local n_sentence = tonumber(
   pg:fetchOne(
      "SELECT MAX(sentence_id) FROM bw.%s_sentence ", {opt.dataset}
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
            function(k, str) return tonumber(str) end
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
