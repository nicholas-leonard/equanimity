require 'torch'
require 'image'
require 'xlua'
_ = require 'underscore'

-- TODO:
--- Flatten images only
--- Class are 1-D
--- Multi-class are 2-D

local DataTensor = torch.class("data.DataTensor")

function DataTensor:__init(...)
   local args, self.data, self.axes, sizes 
      = xlua.unpack(
      {...},
      'DataTensor constructor', nil,
      {arg='data', type='torch.Tensor', 
       help='A torch.Tensor with 2 dimensions or more.', req=true},
      {arg='axes', type='table', 
       help=[[A table defining the order and nature of each dimension
            of a tensor. Two common examples would be the archtypical 
            MLP input : {'b', 'f'}, or a common image representation : 
            {'b', 'h', 'w', 'c'}. 
            Possible axis symbols are :
            1. Standard Axes:
              'b' : Batch/Example
              'f' : Feature
              't' : Class
            2. Image Axes
              'c' : Color/Channel
              'h' : Height
              'w' : Width
              'd' : Dept
            ]], default={'b','f'}},
      {arg='sizes', type='table | torch.LongTensor', 
       help=[[A table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions in axes. This should be supplied 
            if the dimensions of the data is different from the number
            of elements in the axes table, in which case it will be used
            to : data:resize(sizes). Default is data:size().
            ]]}
   )   
   if sizes == nil then
      sizes = self.data:size()
   else
      if type(sizes) == 'table' then
         -- convert table to LongTensor
         sizes = torch.LongTensor(sizes)
      end
      if self.data:dim() ~= #(self.axes) then
         if sizes:size(1) ~= #(self.axes) then
            error("Error: sizes should specify as many dims as axes")
         end
         -- assume data is appropriately contiguous and
         -- convert the data to the default axes format using sizes
         self.data:resize(sizes)
      end
   end
   assert(self.data:dim() == #(self.axes), 
         "Error: data should have as many dims as specified in axes" )
   self.memoized = {}
end


function DataTensor:bf(...)
   local memoize, contiguous = xlua.unpack(
      {...},
      'DataTensor:bf',
      'Returns a 2D-tensor of examples by features'
      {arg='memoize', type='boolean', 
       help='When true caches (memoizes) a contiguous view for later use.', 
       default=false},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   if self.memoized.bf then  
      return self.memoized.bf
   end
   --creates a new view of the same storage
   local data = torch.Tensor(self.data)
   b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim first
      data:transpose(1, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   f = _.indexOf(data, 'f')
   if f == 0 then
      --convert non-b axes to f :
      --reduce tensor down to 2 dimensions: first dim stays the same, 
      --remainder are flattened
      if data:dim() > 2 then
         data:resize(data:size(1), data:size():sub(2,data:dim()):prod())
      end
   end
   if contiguous or memoize then
      data = data:contiguous()
   if memoize then
      self.memoized.bf = data
   return data
end

function DataTensor:bhwc(...)
   local memoize, contiguous, sizes = xlua.unpack(
      {...},
      'DataTensor:bf',
      'Returns a 4D-tensor of examples, by height, by width, by color'
      {arg='memoize', type='boolean', 
       help='When true caches (memoizes) a contiguous view for later use.', 
       default=false},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false},
      {arg='sizes', type='table | torch.Tensor', 
       help=[[A table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions bhwc. This should be supplied 
            if the axes of the DataTensor is different than bhwc, 
            in which case it will be used to : data:resize(sizes) or 
            data:resize(data:size(1), unpack(sizes). The latter requires
            sizes be a table.]]}
   )
   if self.memoized.bhwc then  
      return self.memoized.bhwc
   end
   --creates a new view of the same storage
   local data = torch.Tensor(self.data)
   b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim first
      data:transpose(1, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   if type(sizes) == 'table' then
      if #sizes == 3 then
         sizes = {data:size(1), unpack(sizes)}
      end
      -- convert table to LongTensor
      sizes = torch.LongTensor(sizes)
   end
   if data:dim() == 2 and _.contains(axes, 'f') then
      assert(sizes:size(1) ~= 4, "sizes doesn't have enough dimensions")
      --convert {'b','f'} to {'b','h','w','c'}
      data:resize(sizes)
   else
      error("unsupported conversion of axes formats")
   end
   if contiguous or memoize then
      data = data:contiguous()
   if memoize then
      self.memoized.bf = data
   return data
end


------------------------------------------------------------------------
-- ImageTensor : A DataTensor holding a tensor of images.
------------------------------------------------------------------------
local ImageTensor = torch.class("data.ImageTensor", "data.DataTensor")

function ImageTensor:__init(...)
   args, self.data, self.axes --, self.view_converter
      = xlua.unpack(
      {...},
      'ImageTensor constructor', nil,
      {arg='data', type='table', 
       help=[[Data taking the form of torch.Tensor with 2 dimensions 
            or more. The first dimension be for the number of examples.
            ]], req=true},
      {arg='axes', type='table', 
       help=[[A table defining the order and nature of each dimension
            of a batch of images. An example would be {'b', 0, 1, 'c'}, 
            where the dimensions represent a batch of examples :'b', 
            the first horizontal axis of the image : 0, the vertical 
            axis : 1, and the color channels : 'c'.
            ]], default={'b', 0, 1, 'c'}},
   )
   parent.__init(self, data)
end

------------------------------------------------------------------------
-- ClassTensor : A DataTensor holding a tensor of classes.
------------------------------------------------------------------------
local ClassTensor = torch.class("data.ClassTensor", "data.DataTensor")

function ClassTensor:__init(...)

   parent.__init(self, data)
end

