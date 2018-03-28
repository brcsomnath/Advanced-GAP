# Training a network on MNIST ->
## Testing accuracy
Overall correct 9878 percentage correct98.78 % 	
0	98.877551020408 % 	
1	99.383259911894 % 	
2	99.22480620155 % 	
3	99.009900990099 % 	
4	99.49083503055 % 	
5	98.654708520179 % 	
6	98.12108559499 % 	
7	98.929961089494 % 	
8	99.17864476386 % 	
9	96.828543111992 % 

## Network architecture ->
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
  (1): nn.SpatialConvolution(1 -> 6, 5x5, 1,1, 2,2)
  (2): nn.ReLU
  (3): nn.SpatialMaxPooling(2x2, 2,2)
  (4): nn.SpatialConvolution(6 -> 16, 5x5, 1,1, 2,2)
  (5): nn.ReLU
  (6): nn.SpatialMaxPooling(2x2, 2,2)
  (7): nn.SpatialConvolution(16 -> 32, 3x3, 1,1, 1,1)
  (8): nn.ReLU
  (9): nn.SpatialMaxPooling(2x2, 2,2)
  (10): nn.View(288)
  (11): nn.Linear(288 -> 10)
  (12): nn.LogSoftMax
}
---
