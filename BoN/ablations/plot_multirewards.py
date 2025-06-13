import json
import matplotlib.pyplot as plt

with open('./multireward_uncond.json') as json_data:
    data_uncond = json.load(json_data)
    json_data.close()
    
exp_rew_1_uncond = []
exp_rew_2_uncond = []

for i in data_uncond.keys():
    exp_rew_1_uncond.append(data_uncond[i]['exp_rew_1'])
    exp_rew_2_uncond.append(data_uncond[i]['exp_rew_2'])
    
with open('./multireward_code4.json') as json_data:
    data_code4 = json.load(json_data)
    json_data.close()
    
exp_rew_1_code4 = []
exp_rew_2_code4 = []

for i in data_code4.keys():
    exp_rew_1_code4.append(data_code4[i]['exp_rew_1'])
    exp_rew_2_code4.append(data_code4[i]['exp_rew_2'])
    
with open('./multireward_code40.json') as json_data:
    data_code40 = json.load(json_data)
    json_data.close()
    
exp_rew_1_code40 = []
exp_rew_2_code40 = []

for i in data_code40.keys():
    exp_rew_1_code40.append(data_code40[i]['exp_rew_1'])
    exp_rew_2_code40.append(data_code40[i]['exp_rew_2'])
    
with open('./multireward_freedom.json') as json_data:
    data_freedom = json.load(json_data)
    json_data.close()
    
exp_rew_1_freedom = []
exp_rew_2_freedom = []

for i in data_freedom.keys():
    exp_rew_1_freedom.append(data_freedom[i]['exp_rew_1'])
    exp_rew_2_freedom.append(data_freedom[i]['exp_rew_2'])
    
with open('./multireward_codex.json') as json_data:
    data_codex = json.load(json_data)
    json_data.close()
    
exp_rew_1_codex = []
exp_rew_2_codex = []

for i in data_codex.keys():
    exp_rew_1_codex.append(data_codex[i]['exp_rew_1'])
    exp_rew_2_codex.append(data_codex[i]['exp_rew_2'])
    
plt.scatter(exp_rew_2_uncond,exp_rew_1_uncond,marker='x',label='uncond',color='black')
plt.scatter(exp_rew_2_code4,exp_rew_1_code4,marker='o',label='CoDe (N=4)',color='orange')
plt.scatter(exp_rew_2_code40,exp_rew_1_code40,marker='o',label='CoDe (N=40)',color='green')
plt.scatter(exp_rew_2_freedom,exp_rew_1_freedom,marker='o',label='FreeDoM',color='red')
plt.scatter(exp_rew_2_codex,exp_rew_1_codex,marker='o',label='CoDeX',color='blue')
plt.ylabel("Aesthetic Score")
plt.xlabel("PickScore")  
plt.grid()
plt.legend()
plt.title("Multireward Aesthetic v/s Pickscore")
plt.savefig('multireward.jpg')