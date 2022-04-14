import csv

correct = 0
total = 0
file1 = './Results/BiLSTM_Pre-Train_rst.csv'
with open(file1, 'r') as f1:
    data1 = f1.readlines()
    with open('./data/Task/groundtruth.csv', 'r') as f2:
        data2 = f2.readlines()

for i, line in enumerate(data1):
    if i==0:
        continue
    id1, item1 = line.split(',')
    id2, item2 = data2[i].split(',')
    # if int(id1)+1 != i:
    #     print('id1={}, i={}'.format(id1, i))
    #     break
    # if int(id2)+1 != i:
    #     print('id2={}, i={}'.format(id2, i))
    #     break
    if(item1 == item2):
        correct += 1
    else:
        print(i+1, item1, item2)
    total += 1

print('Acc = {:.6f}'.format(correct/total))