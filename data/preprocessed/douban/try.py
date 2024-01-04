# 1. trust.txt
# list_ = []
# with open('trusts.txt', 'r') as fn:
#     for lines in fn:
#         line = lines.strip().split()
#         list_.append(line)
# with open('trust.txt', 'w') as fn:
#     fn.write('user,friend\n')
#     for a,b in list_:
#         fn.write(a+','+b+'\n')

#2. rating.txt
# import random
# random.seed(2020)
# score = []
# with open('rating.txt', 'r') as fn:
#     for lines in fn:
#         score.append(lines.strip().split())
# random.shuffle(score)
# num_train = int(0.8 * len(score))
#
# with open('train_set.txt', 'w') as fn:
#     fn.write('user,item\n')
#     for user,item in score[:num_train]:
#         fn.write(user+','+item+'\n')
#
# with open('test_set.txt', 'w') as fn:
#     fn.write('user,item\n')
#     for user,item in score[num_train:]:
#         fn.write(user+','+item+'\n')

map_ = {}
map_item = {}
count = 0
count_item = 0
store = []
with open('rating.txt', 'r') as fn:
    for lines in fn:
        line = lines.strip().split()
        store.append(line)
        if line[0] not in map_:
            map_[line[0]] = count
            count += 1
        if line[1] not in map_item:
            map_item[line[1]] = count_item
            count_item += 1

score = [[str(map_[user]), str(map_item[item])] for user,item in store]

with open('ratings.txt', 'w') as fn:
    for user, item in score:
        fn.write(user+' '+item+'\n')

import random
random.seed(2020)
random.shuffle(score)
count_dict = {}
for user,item in score:
    if user in count_dict:
        count_dict[user] += 1
    else:
        count_dict[user] = 1
num_test = int(0.2 * len(score))

train_set = []
test_set = []
total_test = 0
for user,item in score:
    if total_test >= num_test:
        train_set.append([user, item])
    else:
        if count_dict[user] > 1:
            test_set.append([user, item])
            total_test += 1
            count_dict[user] -= 1
        else:
            train_set.append([user, item])
print(len(train_set))
print(len(test_set))
print(len(train_set) / len(test_set))
print(len(train_set) + len(test_set) - len(score))



with open('train_set.txt', 'w') as fn:
    fn.write('user,item\n')
    for user,item in train_set:
        fn.write(user+','+item+'\n')

with open('test_set.txt', 'w') as fn:
    fn.write('user,item\n')
    for user,item in test_set:
        fn.write(user+','+item+'\n')

list_ = []
with open('trusts.txt', 'r') as fn:
    fn.readline()
    for lines in fn:
        line = lines.strip().split(',')
        list_.append(line)
with open('trust.txt', 'w') as fn:
    fn.write('user,friend\n')
    for a,b in list_:
        fn.write(str(map_[a])+','+str(map_[b])+'\n')
