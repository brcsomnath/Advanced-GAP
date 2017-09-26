require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';

model = torch.load('model_MNIST2.t7')

model2 = nn.Sequential()
model2:insert(model:get(13),1)
model:remove(13)
model2:insert(model:get(12),1)
model:remove(12)
model2:insert(model:get(11),1)
model:remove(11)
model2:insert(model:get(10),1)
model:remove(10)
model2:insert(model:get(9),1)
model:remove(9)
model2:insert(model:get(8),1)
model:remove(8)
model2:insert(model:get(7),1)
model:remove(7)

print(model)
print(model2)

checkOut = model:forward(torch.rand(1,28,28))
numFilters = checkOut:size(1)

model_poolAvg = nn.Sequential()
model_poolAvg:add(nn.SpatialAveragePooling(checkOut:size(2),checkOut:size(3)))

trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = mnist.traindataset().label+1
trSize = mnist.traindataset().size

sorted,indices = torch.sort(trainlabels)
classes = {'0', '1', '2','3', '4','5', '6','7', '8','9'}
classSize = torch.Tensor(#classes):zero()
for i=1,trSize do
	classSize[trainlabels[i]] = classSize[trainlabels[i]]+1
end
print(classSize)
print(classSize:sum())

criterion = nn.ClassNLLCriterion()

WeightsTensor = torch.Tensor(numFilters,#classes):zero()
AllClassWeightsTensor = torch.Tensor(numFilters):zero()

batchSize = 200
indexDone=0
errorOccurred = 0

for c=1,#classes do
	print('Starting Class '..c)
	for t=1,classSize[c]/batchSize do
		local input = torch.Tensor(batchSize,1,28,28)
		local target = torch.Tensor(batchSize)
		classTarget = c
		for t1 = 1,batchSize do
			local t2 = (t-1)*batchSize + t1 + indexDone
			input[t1] = trainData[indices[t2]]
			target[t1] = trainlabels[indices[t2]]
			if target[t1]~=c then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward(input)
		local filters_avg = model_poolAvg:forward(filters)
		local filters_avg_sum = torch.Tensor(numFilters)
		for f=1,numFilters do
			collectgarbage()
			filters_avg_sum[f] = filters_avg[{{},{f},{1},{1}}]:sum()
			WeightsTensor[f][c] = WeightsTensor[f][c] + filters_avg_sum[f]
			--print('f '..f..': val '..filters_avg_sum[f])
		end
	end
	if errorOccurred==1 then 
		break 
	end
	if classSize[c]%batchSize~=0 then
		local remaining = classSize[c]%batchSize
		local input = torch.Tensor(remaining,1,28,28)
		local target = torch.Tensor(remaining)
		classTarget = c
		for t1 = 1,remaining do
			local t2 = classSize[c]-remaining + t1 + indexDone
			input[t1] = trainData[indices[t2]]
			target[t1] = trainlabels[indices[t2]]
			if target[t1]~=c then
				print('error at c='..c..' t='..t..' t1='..t1)
				errorOccurred=1
				break;
			end
		end
		if errorOccurred==1 then 
			break 
		end
		local filters = model:forward(input)
		local filters_avg = model_poolAvg:forward(filters)
		local filters_avg_sum = torch.Tensor(numFilters)
		for f=1,numFilters do
			collectgarbage()
			filters_avg_sum[f] = filters_avg[{{},{f},{1},{1}}]:sum()
			WeightsTensor[f][c] = WeightsTensor[f][c] + filters_avg_sum[f]
		end
	end
	if errorOccurred==1 then 
		break 
	end
	indexDone = indexDone+classSize[c]
	for f=1,numFilters do
		AllClassWeightsTensor[f] = AllClassWeightsTensor[f] + WeightsTensor[f][c]
		WeightsTensor[f][c] = WeightsTensor[f][c]/classSize[c]
	end
	--print(errorTensor)
end
AllClassWeightsTensor = AllClassWeightsTensor/trSize

torch.save('newerGAP_WeightsTensor.t7',WeightsTensor)
torch.save('newerGAP_WeightsTensor_allClass.t7',AllClassWeightsTensor)