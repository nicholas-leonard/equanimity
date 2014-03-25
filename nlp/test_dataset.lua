require 'dp'

local pg = dp.Postgres()

local sentence_set = {}
local size = 0

local n_sentence = tonumber(pg:fetchOne("SELECT MAX(sentence_id) FROM bw.train_sentence")[1])
local step = 1000
local start_time = os.time()
print("loading sentences from db")
for i=1,n_sentence,step do
   local rows = pg:fetch(
      "SELECT sentence_id, sentence_words " ..
      "FROM bw.train_sentence " ..
      "WHERE sentence_id BETWEEN %d and %d", 
      {i, i+step-1}
   )
   for j, row in ipairs(rows) do
      local sentence_id = tonumber(row[1])
      local sentence_words = torch.IntTensor(
         _.map(_.split(string.sub(row[2], 2, -2), ','), function(str) return tonumber(str) end)
      )
      sentence_set[sentence_id] = sentence_words
      size = size + sentence_words:size(1) - 1
   end
   xlua.progress(i, n_sentence)
end
print("\nloaded " .. n_sentence .. " sentences in " .. os.time()-start_time .. " sec.")

local data_set = torch.IntTensor(size, 2):zero()
local sentence_idx = 1
local i = 0
start_time = os.time()
for sentence_id, sentence_words in pairs(sentence_set) do
   local sentence_size = sentence_words:size(1)
   local sentence_slice = data_set:narrow(1, sentence_idx, sentence_size - 1)
   sentence_slice:select(2,1):fill(sentence_id)
   sentence_slice:select(2,2):copy(sentence_words:sub(2,sentence_size))
   --print(sentence_idx, sentence_slice:size(1), data_set:size(1), sentence_size-1)
   sentence_idx = sentence_idx + sentence_size - 1
   i = i + 1
   xlua.progress(i, n_sentence)
end
print("indexed " .. n_sentence .. " sentences in " .. os.time()-start_time .. " sec.")

--print(data_set[{{data_set:size(1)-100, data_set:size(1)},{}}])
