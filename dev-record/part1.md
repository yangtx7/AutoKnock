1.初始化工具箱 [0:00-2:00]
> initCobraToolbox

2.模型读入（.mat文件）[2:00-3:00]
> model=readCBModel('yeast-GEM.mat')

3.格式转换（.mat文件->.excel文件）[3:00-4:00]
> writeCBModel(model,'xlsx','yeast-GEM.xlsx)
> 格式转换的目的是方便后续手动在excel文件中添加、修改数据

4.yeast-GEM.xlsx文件中各变量解释 [4:00-6:00]
> 格式转换完的yeast-GEM.xlsx文件中包含两个sheet，分别是Reaction List 和 Metabolite List
> （1）需要在两个List最下方添加代谢物、代谢反应信息
> （2）针对一些反应修改模型参数 
>   注意：具体补充的信息和补充的规则视频中未提及，可能还需要进一步确认

5.读取Yeast8-0A07-.xlsx文件（即针对yeast-GEM.xlsx添加、修改后的文件）[6:00-7:40]
> model=xls2model('Yeast8-0A07-.xlsx')

6.FBA计算 [7:40-9:00]
> FBAsolution=optimizeCBModel(model)
> FBA计算的是所有反应物的代谢通量值
> 将计算出的FBA结果手动添加到'Yeast8-0A07-.xlsx'文件的最后一列
> FBA计算的目的是作为Optknock计算迭代的终止条件依据

7.OptKnock计算  [9:00-17:49]
> （1）初始化一个代学反映集: selectedRxnList={[1]}  [9:00-9:50]
> （2）把所有反应物（方案）ID手动添加到代学反映集中    [9:50-11:00]
> （3）OptKnock计算  [11:00-13:00]
    selectedRxnList={'r_0005';'r_0006';'r_0018}  #填入所有的待选反应
    fbaWT = optimizeCbModel(model)  #FBA计算
    options.targetRxn='r_4693'  #指定目标方程，根据具体模型填入
    options.vmax=1000 
    options.numDel=1  #设定基因敲除数，根据计算需求设定值
    options.numDelSense='L' 
    constrOpt.rxnList={'r_2111','r_4046'}  #根据具体模型填入
    constrOpt.values=[0.5*fbaWT.f,0.7]  #会使用FBA计算值为一个约束
    constrOpt.sense='GE' 
    OptKnockSol=OptKnock(model,selectedRxnList,options,constrOpt)
> （4）记录OptKnock计算结果  [13:00-14:20]
> 需要记录的结果有两个：得到的反应物（方案）ID和全局反应通量值，添加到Yeast8-0A07-.xlsx的最后一列，即FBA结果后一列
> （5）迭代计算  [14:20-16:30]
> 把上一步得到的方案（反应物ID）移除，继续计算
> 如果上一步得到的反应物ID是'r_0006'，则这次输入的命令需要把'r_0006'从selectedRxnList中移除，其余保持不变
> （6）终止条件  [16:30-17:49]
> 终止条件是连续两次OptKnock计算得到的通量值小于FBA计算得到的结果