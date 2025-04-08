
import itertools
import networkx as nx
import pandas as pd
from cobra.io import load_model,load_matlab_model
from scipy.io import savemat
import matlab.engine



# 读取两个currency metabolites的CSV文件
pairs_of_currency_metabolites_file = "data/pairs_of_currency_metabolites.csv"
special_currency_metabolites_file = "data/special_currency_metabolites.csv"
pairs_of_currency_metabolites = pd.read_csv(pairs_of_currency_metabolites_file)
special_currency_metabolites = pd.read_csv(special_currency_metabolites_file)

# 处理currency metabolites
pi_pairs1 = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['pi_pairs1'] if str(x) != 'nan']
pi_pairs2 = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['pi_pairs2'] if str(x) != 'nan']
h_pairs1 = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['h_pairs1'] if str(x) != 'nan']
h_pairs2 = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['h_pairs2'] if str(x) != 'nan']
nh4_pairs = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['nh4_pairs'] if str(x) != 'nan']
other_pairs = [tuple(x.split(",")) for x in pairs_of_currency_metabolites['other_pairs'] if str(x) != 'nan']

excluded = [x for x in special_currency_metabolites['excluded'] if str(x) != 'nan']
nocarbon = [x for x in special_currency_metabolites['no carbon metabolites'] if str(x) != 'nan']
ions = [x for x in special_currency_metabolites['ions'] if str(x) != 'nan']

currency_mets = excluded + nocarbon + ions

# 定义函数
def get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets):
    # get the metabolite links for a reaction, excluding links through currency metabolites
    # processing in the order of P & H transfer, N transfer and other transfers
    sub_pro = []
    mark = 0  # mark if there is currency metabolite pairs for P/H transfer in the reaction
    c2mark = 0  # mark if there is currency metabolite pairs for P/H transfer in the reaction, second batch
    nmark = 0  # mark if there is nh4 transfer currency metabolite pairs in the reaction, deal seperately
    omark = 0  # mark if there is other group transfer currency metabolite pairs in the reaction, deal seperately
    cmet = []  # temorary currency metabolite list
    c2met = []  # temorary currency metabolite list, second batch
    ncmet = []  # temorary currency metabolite list for N transfer
    ocmet = []  # temorary currency metabolite list for other pair transfer
    phmet = []  # metabolite for P/H transfer, to be excluded again in the last step if >1 pairs still remaining, especially for UDP ADPsugars
    ex_pairs1 = pi_pairs1+h_pairs1  # excluded currency metabolite pairs, first batch
    ex_pairs2 = pi_pairs2+h_pairs2  # excluded currency metabolite pairs, second batch
    for sp in ex_pairs1:
        phmet.append(sp[0])
        phmet.append(sp[1])
    for sp in ex_pairs2:
        phmet.append(sp[0])
        phmet.append(sp[1])
    phmet = list(set(phmet))  # remove repeats
    # also check if CoA is a currency metabolite in the last step
    phmet += ['s_0529[c]', 's_0530[w]', 's_0531[a]', 's_0532[m]', 's_0533[n]', 's_0534[p]', 's_2785[x]', 's_2857[q]', 's_3321[h]']
    subs = [m.id for m in rea.reactants if m.id not in currency_mets]
    pros = [m.id for m in rea.products if m.id not in currency_mets]
    for s, p in itertools.product(subs, pros):
        if (s, p) in ex_pairs1 or (p, s) in ex_pairs1:  # need to consider direction
            mark = 1
            cmet.append(s)
            cmet.append(p)
        if (s, p) in ex_pairs2 or (p, s) in ex_pairs2:  # need to consider direction
            c2mark = 1
            c2met.append(s)
            c2met.append(p)
        if (s, p) in nh4_pairs or (p, s) in nh4_pairs:  # for nh4 transfer
            nmark = 1
            ncmet.append(s)
            ncmet.append(p)
        if (s, p) in other_pairs or (p, s) in other_pairs:  # for other pair transfer
            omark = 1
            ocmet.append(s)
            ocmet.append(p)
    # if len(sub_pro)>1 and mark==1:
    if mark == 1:  # process in order
        subs = [m for m in subs if m not in cmet]
        pros = [m for m in pros if m not in cmet]
    if c2mark == 1:  # process in order
        subsn = [m for m in subs if m not in c2met]
        prosn = [m for m in pros if m not in c2met]
        if subsn and prosn:  # proceed if only there are still other metabolites in the reactant and product list
            subs = subsn
            pros = prosn
    if nmark == 1:
        subsn = [m for m in subs if m not in ncmet]
        prosn = [m for m in pros if m not in ncmet]
        if subsn and prosn:  # proceed if only there are still other metabolites in the reactant and product list
            subs = subsn
            pros = prosn
    if omark == 1:
        subsn = [m for m in subs if m not in ocmet]
        prosn = [m for m in pros if m not in ocmet]
        if subsn and prosn:  # proceed if only there are still other metabolites in the reactant and product list
            subs = subsn
            pros = prosn
    if len(subs) > 1:  # to remove UTP, UDP etc.
        subsn = subs
        for m in subsn:
            if m in phmet:
                subs.remove(m)
                if len(subs) == 1:
                    break
    if len(pros) > 1:  # to remove UTP, UDP etc.
        prosn = pros
        for m in prosn:
            if m in phmet:
                pros.remove(m)
                if len(pros) == 1:
                    break
    for s, p in itertools.product(subs, pros):
        sub_pro.append((s, p))
    return sub_pro

# # 读取模型
# from cobra.io import load_model,load_matlab_model
# model = load_matlab_model("data\Yeast8-OA.mat")
# # 读取各反应的代谢通量值(经由OptKnock计算后得到的fluxes值)
# flux_value_file = 'data/test1v_valuesoptknock.xlsx'
# flux_values_df = pd.read_excel(flux_value_file)
# # 根据列名称'Abbreviation'获取反应ID
# reaction_ids = flux_values_df['Abbreviation']
# # 根据列名称'。。。'获取通量值列
# flux_values_series = flux_values_df.iloc[:, 10]  #对应的通量值所在的列名
# # 将反应ID和对应的通量值存入字典
# flux_values = dict(zip(reaction_ids, flux_values_series))
# # 设置通量值的阈值
# flux_threshold = 1e-10
# # 遍历模型中的反应，并更新通量值
# for rea in model.reactions:
#     # 根据反应ID获取通量值，如果不存在则默认为0
#     flux_value = flux_values.get(rea.id, 0)
#     # 如果通量值的绝对值小于阈值，则将其设为0
#     if abs(flux_value) < flux_threshold:
#         flux_value = 0
#     # 将通量值赋给模型中的反应对象
#     rea.flux_value = flux_value
# # 创建有向图
# G = nx.MultiDiGraph()
# # 遍历模型中的反应，添加到图中
# for rea in model.reactions:
#     if not rea.boundary:
#         sub_pro = get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets)
#         # 检查反应的通量值，仅添加不为0的反应
#         flux_value = rea.flux_value
#         if flux_value != 0:
#             # 对于正向通量，添加从反应物到产物的边
#             if flux_value > 0:
#                 for sp in sub_pro:
#                     G.add_edge(sp[0], sp[1], label=rea.id, weight=1)  # 反应物指向产物,如果是用实际通量值为权重值weight=abs(flux_value)
#             # 对于负向通量且反应可逆，添加从产物到反应物的边
#             elif rea.reversibility:
#                 for sp in sub_pro:
#                     G.add_edge(sp[1], sp[0], label=rea.id, weight=1)  # 产物指向反应物
# #将图写入CSV文件
# result_save_path = './'
# edgelist = []
# for u, v, d in G.edges(data=True):
#     edgelist.append([u, v, d['label'], d['weight']])
# edgelist_df = pd.DataFrame(edgelist, columns=['source', 'target', 'reaction', 'weight'])
# edgelist_df.to_csv(result_save_path + '-Δr_0172_edgelist.csv', index=False)

# import matplotlib.pyplot as plt
# # 绘制图形
# def draw_graph(G):
#     nx.draw(G, with_labels=True, arrows=True)
#     plt.show()
# # 调用绘制函数
# draw_graph(G)



import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from cobra.io import load_matlab_model

# 读取模型
model = load_matlab_model("data/Yeast8-OA.mat")

# 读取各反应的代谢通量值
flux_value_file = 'data/test1v_valuesoptknock.xlsx'
flux_values_df = pd.read_excel(flux_value_file)

# 根据列名称'Abbreviation'获取反应ID
reaction_ids = flux_values_df['Abbreviation']
flux_values_series = flux_values_df.iloc[:, 10]  # 对应的通量值所在的列名
flux_values = dict(zip(reaction_ids, flux_values_series))

# 设置通量值的阈值
flux_threshold = 1e-10

# 遍历模型中的反应，并更新通量值
for rea in model.reactions:
    flux_value = flux_values.get(rea.id, 0)
    if abs(flux_value) < flux_threshold:
        flux_value = 0
    rea.flux_value = flux_value

# 创建有向图
G = nx.MultiDiGraph()

# 遍历模型中的反应，添加到图中
for rea in model.reactions:
    if not rea.boundary:
        # 假设 get_metpair 函数已定义
        sub_pro = get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets)
        flux_value = rea.flux_value
        if flux_value != 0:
            if flux_value > 0:
                for sp in sub_pro:
                    G.add_edge(sp[0], sp[1], label=rea.id, weight=abs(flux_value))  # 使用通量值作为权重
            elif rea.reversibility:
                for sp in sub_pro:
                    G.add_edge(sp[1], sp[0], label=rea.id, weight=abs(flux_value))

# 使用 Plotly 绘制拓扑图
def draw_plotly_graph(G):
    pos = nx.spring_layout(G)  # 计算节点位置
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # None用于断开线条
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
        node_text.append(f'{node}<br># of connections: {len(list(G.neighbors(node)))}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='<br>Network Graph made with Python',
                        
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )

    fig.write_image(path)


# 调用绘制函数
draw_plotly_graph(G)

import networkx as nx
import pandas as pd
# 假设 G 是已经创建好的图
# 定义两个不同的目标节点
target1 = 's_0450[c]'  #biomass节点
target2 = 's_4422[e]'  #product节点
# 读取源节点的csv文件，假设csv文件中有名为 'Abbreviation(in GEM)' 的列
source_file = 'data\TF-based Biosensors.xlsx'
# 使用 pd.ExcelFile 读取 Excel 文件，得到 ExcelFile 对象
excel_file = pd.ExcelFile(source_file)
# 调用 parse 方法获取指定工作表的数据，并设置表头行为 1
sources_df = excel_file.parse('Sheet1', header=1)
# 结果存储列表
results = []
# 对每个源节点计算到两个目标节点的最短路径长度，以及比值
for index, row in sources_df.iterrows():
    source = row['Abbreviation(in GEM)']
    # 初始化最短路径长度为无穷大（表示无路径）
    shortest_path_length_to_target1 = float('inf')
    shortest_path_length_to_target2 = float('inf')
    # 计算从源节点到第一个目标节点的最短路径长度
    if source in G and target1 in G:
        try:
            shortest_path_length_to_target1 = nx.shortest_path_length(G, source, target1)
        except nx.NetworkXNoPath:
            print(f"从 {source} 到 {target1} 没有路径。")
    # 计算从源节点到第二个目标节点的最短路径长度
    if source in G and target2 in G:
        try:
            shortest_path_length_to_target2 = nx.shortest_path_length(G, source, target2)
        except nx.NetworkXNoPath:
            print(f"从 {source} 到 {target2} 没有路径。")
    # 计算比值，如果到 target2 的最短路径长度为无穷大，则比值设为 0
    ratio = shortest_path_length_to_target1 / shortest_path_length_to_target2 if shortest_path_length_to_target2 != float('inf') else 0
    # 存储结果
    results.append({
        'source_id': source,
        'shortest_path_length_to_target1': shortest_path_length_to_target1,
        'shortest_path_length_to_target2': shortest_path_length_to_target2,
        'ratio': ratio
    })

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)
# 按照比值从小到大排序结果DataFrame
sorted_results_df = results_df.sort_values(by='ratio')
# 将排序后的结果写入xlsx文件
output_file = 'Δr_0172_sorted_results.xlsx'
sorted_results_df.to_excel(output_file, index=False)
print(f"计算结果已写入 {output_file}，并按比值从小到大排序。")
