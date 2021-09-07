import torch

# class_num = 10
# batch_size = 5
# label = torch.LongTensor(batch_size, 1).random_() % class_num
# one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
# print(one_hot)
li = ['music', 'middleeast', 'sports', 'travel', 'health', 'foodanddrink', 'news', 'autos', 'video', 'tv', 'kids', 'games', 'finance', 'lifestyle', 'movies', 'northamerica', 'entertainment', 'weather']
num = 18
x = torch.zeros(num)
x[(li.index('weather')) % num] = 1
print(x)
