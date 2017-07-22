require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(1,96,3,3,1,1,2,2)) --1
model:add(nn.ReLU()) --2
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --3
-- size: 96X15X15

model:add(nn.SpatialConvolution(96,256,3,3,1,1,2,2)) --4
model:add(nn.ReLU()) --5
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --6
-- size: 256X8X8

model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1)) --7
model:add(nn.ReLU()) --8
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --9
-- size: 256X4X4

model:add(nn.View(-1):setNumInputDims(3)) --10
model:add(nn.Linear(4096,512)) --11
model:add(nn.Linear(512,10)) --12
model:add(nn.Sigmoid()) --13
model:add(nn.LogSoftMax()) --14

return model
