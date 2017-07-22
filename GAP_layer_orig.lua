require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = torch.load('model_MNIST2.t7')

model:remove(14)
model:remove(13)
model:remove(12)
model:remove(11)
model:remove(10)

checkOut = model:forward(torch.rand(1,28,28))
numFilters = checkOut:size(1)

model2 = nn.Sequential()
model2:add(nn.Linear(256,10))
model2:add(nn.LogSoftMax())

print(model)
print(model2)
print('Number of Filters: '..numFilters)

trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = mnist.traindataset().label+1
trSize = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28)
testlabels = mnist.testdataset().label+1
teSize = mnist.testdataset().size
print(trSize,teSize)

criterion = nn.ClassNLLCriterion()

local trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.02
trainer.learningRateDecay = 0.001
trainer.shuffleIndices = 0
trainer.maxIteration = 30
batchSize = 300;

collectgarbage()
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.Tensor(batchSize,1,28,28);
local filter_output=torch.Tensor(batchSize,numFilters);
local target=torch.Tensor(batchSize);
local errorTensor = {}
print(trSize, trSize/batchSize);
print("Training starting")


while true do
	local currentError_ = 0
    for t = 1,math.floor(trSize/batchSize) do
    	local currentError = 0;
      	for t1 = 1,batchSize do
      		t2 = (t-1)*batchSize+t1;
        	target[t1] = trainlabels[t2];
        	input[t1] = trainData[t2];
        	local temp_out = model:forward(input[t1])
        	for filter_iter=1,numFilters do
        		filter_output[t1][filter_iter] = temp_out[filter_iter]:mean()
        	end
			--print(t1)
        end
        currentError = currentError + criterion:forward(model2:forward(filter_output), target)
        --print(currentError)
		currentError_ = currentError_ + currentError*batchSize;
 		model2:updateGradInput(filter_output, criterion:updateGradInput(model2:forward(filter_output), target))
 		model2:accUpdateGradParameters(filter_output, criterion.gradInput, currentLearningRate)
 		if (t%1==0) then 
 			print("batch "..t.." done ==>"); 
 		end
 		collectgarbage()
    end
    ---- training on the remaining images, i.e. left after using fixed batch size.
    if(trSize%batchSize ~=0) then
	    local residualInput = torch.Tensor(trSize%batchSize,1,28,28);
	    local residualFilter_output = torch.Tensor(trSize%batchSize,numFilters);
	    local residualTarget = torch.Tensor(trSize%batchSize);

	    for t1=1,(trSize%batchSize) do
	    	t2=batchSize*math.floor(trSize/batchSize) + t1;
	    	residualTarget[t1] = trainlabels[t2];
	    	residualInput[t1] = trainData[t2];
	    	local temp_out = model:forward(residualInput[t1])
        	for filter_iter=1,numFilters do
        		residualFilter_output[t1][filter_iter] = temp_out[filter_iter]:mean()
        	end
		end
		currentError_ = currentError_ + criterion:forward(model2:forward(residualFilter_output), residualTarget)*(trSize%batchSize)
		--print("_ "..currentError_);
 		model2:updateGradInput(residualFilter_output, criterion:updateGradInput(model2:forward(residualFilter_output), residualTarget))
 		model2:accUpdateGradParameters(residualFilter_output, criterion.gradInput, currentLearningRate)
 		collectgarbage()
	end
	currentError_ = currentError_ / trSize
	print("#iteration "..iteration..": current error = "..currentError_);
	errorTensor[iteration] = currentError_;
	iteration = iteration + 1
  	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
  	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
    	print("# StochasticGradient: you have reached the maximum number of iterations")
     	print("# training error = " .. currentError_)
     	break
  	end
  	collectgarbage()
end

print(errorTensor)

print('Testing accuracy')
correct = 0
class_perform = {0,0,0,0,0,0,0,0,0,0}
class_size = {0,0,0,0,0,0,0,0,0,0}
classes = {'0', '1', '2','3', '4','5', '6','7', '8','9'}
for i=1,teSize do
    local groundtruth = testlabels[i]
    local example = torch.Tensor(1,28,28);
    example = testData[i]
    local filters = torch.Tensor(numFilters)
    local temp = model:forward(example)
    for filter_iter = 1,numFilters do
    	filters[filter_iter] = temp[filter_iter]:mean()
    end
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model2:forward(filters)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end

torch.save('gapWeights_MNIST2.t7',model2)