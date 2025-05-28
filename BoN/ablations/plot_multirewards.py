import json
import matplotlib.pyplot as plt

with open('./multireward.json') as json_data:
    data = json.load(json_data)
    json_data.close()
    
exp_rew_1 = []
exp_rew_2 = []

for i in data.keys():
    exp_rew_1.append(data[i]['exp_rew_1'])
    exp_rew_2.append(data[i]['exp_rew_2'])
    
plt.scatter(exp_rew_2,exp_rew_1,marker='o')
plt.ylabel("Aesthetic Score")
plt.xlabel("PickScore Score")  
plt.grid()
plt.title("Multireward Aesthetic v/s Pickscore")
plt.savefig('multireward.jpg')