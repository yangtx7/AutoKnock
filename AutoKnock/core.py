import matlab.engine
import pandas as pd
import shutil
import os
from openpyxl import load_workbook
import networkx as nx
from cobra.io import load_matlab_model
from init_parameter import *
from utils import read_model,add_v_to_excel_newfile,add_optknock_results_to_excel
from methods import FBA_compute,perform_optknock,check_termination_condition,Handling_Currency_Metabolites,get_metpair,draw_plotly_graph



def full_pipeline(args):
    """
    Executes the full optimization pipeline, from initializing the MATLAB engine to performing FBA computations and 
    optimizations using OptKnock.
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Initialize COBRA Toolbox in MATLAB
    eng.eval("initCobraToolbox", nargout=0)
    print("MATLAB Version: ", eng.version())  # Print MATLAB version

    # Read model file
    model_path = read_model(args.model_path, eng)
    
    # Compute flux balance analysis (FBA)
    v_values = FBA_compute(eng)
    
    # Add v values to a new Excel file
    df_v, df_v_path = add_v_to_excel_newfile(model_path, v_values, new_column_name='v_values', suffix='v_values')

    # Reaction list processing (to be implemented)
    # rxnList = transfer_reaction_list(excel_file_path=reaction_list_path)
    
    rxnList = []

    # Copy the original Excel file to create a working copy
    base = os.path.splitext(df_v_path)[0]
    ext = os.path.splitext(df_v_path)[1]
    new_path = f"{base}{'optknock'}{ext}"
    shutil.copyfile(df_v_path, new_path)

    wb = load_workbook(new_path)

    # Read reaction list from the Excel file
    df = pd.read_excel(args.reaction_list_path)
    reaction_list = df.iloc[:, 0].tolist()  # Get the first column as the reaction list
    count = 0
    consecutive_failures = [0]  # 使用列表保持状态，因为整数是不可变的
    flux_values_name_list=[]
    while True:
        count += 1
        OptKnockSol = perform_optknock(reaction_list, eng, args.target_rxn, args.vmax, args.num_del, args.num_del_sense, args.constr_opt_rxn_list, args.constr_opt_values1, args.constr_opt_values2, args.constr_opt_sense)
        rxnList = eng.eval("OptKnockSol.rxnList", nargout=1)
       
        new_col_name = str(rxnList)
        if new_col_name.startswith("['") and new_col_name.endswith("']"):
            new_col_name = new_col_name[2:-2]

        print(new_col_name)
        print(type(new_col_name))
            

        flux_values_name_list.append(new_col_name)
        # Write results to the workbook
        wb = add_optknock_results_to_excel(reaction_list, eng, wb, OptKnockSol,new_col_name)
        
        # Save the updated workbook
        wb.save(new_path)
        print(f"Saved results after {count} iterations")  # Print iteration count

        # Check termination condition (e.g., stop after 9 iterations)
        if check_termination_condition(wb,count,consecutive_failures,eng):

            break
    
    print('start part 2')
    # Part 1 ends, part 2 begins (could include more operations like model editing, generating reports, etc.)
    pi_pairs1,pi_pairs2,h_pairs1,h_pairs2,nh4_pairs,other_pairs,currency_mets=Handling_Currency_Metabolites(args.pairs_of_currency_metabolites_file,args.special_currency_metabolites_file)
   

    # 读取模型
    model = load_matlab_model(args.model_path_mat)

    # 读取各反应的代谢通量值
    
    flux_values_df = pd.read_excel(args.flux_value_file)

    # 根据列名称'Abbreviation'获取反应ID
    reaction_ids = flux_values_df['Abbreviation']
    print(flux_values_name_list)
    print(type(flux_values_name_list))

    for col_name in flux_values_name_list:
        flux_values_series = flux_values_df[col_name]  # 对应的通量值所在的列名
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
        print('创建有向图')
        G = nx.MultiDiGraph()

        # 遍历模型中的反应，添加到图中
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('outputs/pic', exist_ok=True)
        os.makedirs('outputs/table', exist_ok=True)
        for rea in model.reactions:
            if not rea.boundary:
                # 假设 get_metpair 函数已定义
                sub_pro = get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets,args.phmet_list)
                flux_value = rea.flux_value
                if flux_value != 0:
                    if flux_value > 0:
                        for sp in sub_pro:
                            G.add_edge(sp[0], sp[1], label=rea.id, weight=abs(flux_value))  # 使用通量值作为权重
                    elif rea.reversibility:
                        for sp in sub_pro:
                            G.add_edge(sp[1], sp[0], label=rea.id, weight=abs(flux_value))

        # 使用 Plotly 绘制拓扑图
        path=os.path.join('outputs', 'pic', f"{col_name}_graph.png")
        # 调用绘制函数
        print('调用绘制函数')
        draw_plotly_graph(G,path)
        target1 = 's_0450[c]'  #biomass节点
        target2 = 's_4422[e]'  #product节点
        # 读取源节点的csv文件，假设csv文件中有名为 'Abbreviation(in GEM)' 的列
        source_file = os.path.join("data", "TF-based Biosensors.xlsx")
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
        output_file = os.path.join("outputs", "table", f"{col_name}_results.xlsx")
        sorted_results_df.to_excel(output_file, index=False)
        print(f"计算结果已写入 {output_file}，并按比值从小到大排序。")



    # Additional operations can be implemented here, e.g., removing CM, drawing figures, generating reports.

# Example usage
if __name__ == "__main__":

    parser=init_parametes()
    args = parser.parse_args()

    full_pipeline(args)