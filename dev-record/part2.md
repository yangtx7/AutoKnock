## 流程总述：
1.导入流程1修改并补充FBA与optknock结果的模型文件
> .mat

2.读取两个流通代谢物文件（附件2：pairs_of_currency_metabolites.csv和附件3：special_currency_metabolites.csv）,进行流通代谢物移除。
> 函数 get_metpair（）

3.依据optkonk结果的通量值对模型进行处理，optkonk结果的有n个通量值，与之对应计算n个拓扑图、拓扑距离比值。
```
#读取模型
from cobra.io import load_model,load_matlab_model
model = load_matlab_model("./Yeast8-OA.mat")
#读取各反应的代谢通量值(经由OptKnock计算后得到的fluxes值)
flux_value_file = './Yeast8-OA.xlsx'
flux_values_df = pd.read_excel(flux_value_file)
#根据列名称'Abbreviation'获取反应ID
reaction_ids = flux_values_df['Abbreviation']
#根据列名称'。。。'获取通量值列
flux_values_series = flux_values_df['Δr_0172']  #对应的通量值所在的列名
#将反应ID和对应的通量值存入字典
flux_values = dict(zip(reaction_ids, flux_values_series))
#设置通量值的阈值
flux_threshold = 1e-10
#遍历模型中的反应，并更新通量值
for rea in model.reactions:
    # 根据反应ID获取通量值，如果不存在则默认为0
    flux_value = flux_values.get(rea.id, 0)
    # 如果通量值的绝对值小于阈值，则将其设为0
    if abs(flux_value) < flux_threshold:
        flux_value = 0
    # 将通量值赋给模型中的反应对象
    rea.flux_value = flux_value
```
4.创建点线关系有向图
>存储为csv文件，并将拓扑图以图片形式保存

5.拓扑距离计算，读取TF-based biosensor.csv文件主要计算TF-based biosensors.xlsx文件中多个源节点（【Abbreviation(in GEM)】列）到拓扑图中两个指定目标节点的最短路径，并计算路径长度的比值，将比值结果按从小到大排序，计算结果存储进行拓扑距离计算，输出计算结果及排序。 

6.输出结果：
- 每个Optknock计算下的方案(本次计算中每一步的具体输入的文件信息或者参数信息)，包括代谢反应名称、反应关联的基因名称信息；
- 每个方案下经过模型处理及计算后筛选得到的化合物-转录因子方案，主要包括TF-based biosensors.xlsx文件中化合物名称（Compound）、转录因子名称（TF name）信息。相关结果报告可以导出成常见的文件格式。


## 问题汇总

- 不同的模型文件列名、列数不一致，需要进一步确定读取方式
- get_metpair（）函数健壮性待验证
- 结合前面步骤读取存储所有optknock结果通量值的变量，循环得出全部的拓扑图与拓扑距离比值
- 拓扑距离计算中两个节点的定义
> target1 = 's_0450[c]'  #biomass节点 此节点定义规则未明确，暂定为依据模型中biomass关键字定位
  target2 = 's_4422[e]'  #product节点 此节点需要在模型编辑阶段进行输入

- 调研Python中拓扑图可视化方法
