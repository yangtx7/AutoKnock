import matlab.engine
import pandas as pd
import shutil
import os
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers
import itertools
import networkx as nx
import plotly.graph_objects as go
from cobra.io import load_matlab_model

def full_pipeline(model_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense,phmet_list,pairs_of_currency_metabolites_file,special_currency_metabolites_file,model_path_mat,flux_value_file):
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
    model_path = read_model(model_path, eng)
    
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
    df = pd.read_excel(reaction_list_path)
    reaction_list = df.iloc[:, 0].tolist()  # Get the first column as the reaction list
    count = 0
    consecutive_failures = [0]  # 使用列表保持状态，因为整数是不可变的
    flux_values_name_list=[]
    while True:
        count += 1
        OptKnockSol = perform_optknock(reaction_list, eng, df_v_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense)
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
    pi_pairs1,pi_pairs2,h_pairs1,h_pairs2,nh4_pairs,other_pairs,currency_mets=Handling_Currency_Metabolites(pairs_of_currency_metabolites_file,special_currency_metabolites_file)
   

    # 读取模型
    model = load_matlab_model(model_path_mat)

    # 读取各反应的代谢通量值
    
    flux_values_df = pd.read_excel(flux_value_file)

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
        for rea in model.reactions:
            if not rea.boundary:
                # 假设 get_metpair 函数已定义
                sub_pro = get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets,phmet_list)
                flux_value = rea.flux_value
                if flux_value != 0:
                    if flux_value > 0:
                        for sp in sub_pro:
                            G.add_edge(sp[0], sp[1], label=rea.id, weight=abs(flux_value))  # 使用通量值作为权重
                    elif rea.reversibility:
                        for sp in sub_pro:
                            G.add_edge(sp[1], sp[0], label=rea.id, weight=abs(flux_value))

        # 使用 Plotly 绘制拓扑图
        path=os.path.join('\\AutoKnock\\picture', f"{col_name}_graph.png")
        # 调用绘制函数
        print('调用绘制函数')
        draw_plotly_graph(G,path)
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
        output_file = f"{col_name}_results.xlsx"
        sorted_results_df.to_excel(output_file, index=False)
        print(f"计算结果已写入 {output_file}，并按比值从小到大排序。")



    # Additional operations can be implemented here, e.g., removing CM, drawing figures, generating reports.

def read_model(model_path, eng):
    """
    Reads the model from the provided file path and converts it to an Excel file if necessary.
    """
    if model_path.endswith('.mat'):
        eng.eval(f"model = readcbModel('{model_path}')", nargout=0)
        print(f"Model read using readcbModel from .mat file: {model_path}")
        base = os.path.splitext(model_path)[0]
        model_path = f"{base}.xlsx"
        eng.eval(f"writeCBModel(model,'xlsx','{model_path}')", nargout=0)
        
        return model_path

    elif model_path.endswith('.xlsx'):
        eng.eval(f"model = xls2model('{model_path}')", nargout=0)
        print(f"Model read using xls2model from .xlsx file: {model_path}")
        
        return model_path
    else:
        print("Unsupported file type. Please provide .mat or .xlsx files.")

def FBA_compute(eng):
    """
    Computes flux balance analysis (FBA) for the given model.
    """
    eng.eval("FBAsolution = optimizeCbModel(model)", nargout=0)  # Assume 'model' is defined previously
    # Extract v values (flux values)
    v_values = eng.eval("FBAsolution.v", nargout=1)
    print("FBA computation complete. Flux values extracted.")
    return v_values

def add_v_to_excel_newfile(excel_path, v_values, new_column_name='v_values', suffix='_modified'):
    """
    Adds a new column of v values to an Excel file, saving it as a new file.
    """
    # Generate new file path
    base = os.path.splitext(excel_path)[0]
    ext = os.path.splitext(excel_path)[1]
    new_path = f"{base}{suffix}{ext}"

    # Copy original file
    if os.path.exists(new_path):
        os.remove(new_path)
    shutil.copyfile(excel_path, new_path)

    wb = load_workbook(new_path)

    # Check if the sheet 'Reaction List' exists
    if 'Reaction List' not in wb.sheetnames:
        wb.close()
        os.remove(new_path)
        raise ValueError("No 'Reaction List' sheet found in the file.")
    
    ws = wb['Reaction List']
    
    # Validate row count matches
    data_row_count = ws.max_row - 1  # Excluding header row
    if len(v_values) != data_row_count:
        wb.close()
        os.remove(new_path)
        raise ValueError(f"Data row count mismatch (v values: {len(v_values)}, sheet rows: {data_row_count})")

    # Insert v values into a new column
    insert_col = 1
    while insert_col <= ws.max_column:
        cell = ws.cell(row=1, column=insert_col)
        if cell.value is None:
            break
        insert_col += 1
    else:
        insert_col = ws.max_column + 1  # Insert at the end if no empty cells

    headers = [cell.value for cell in ws[1]]
    new_col_name = new_column_name
    counter = 1
    while new_col_name in headers:
        new_col_name = f"{new_column_name}_{counter}"
        counter += 1

    ws.cell(row=1, column=insert_col, value=new_col_name)

    # Write the v values to the new column
    sci_format = '0.#####E+00'
    for idx, value in enumerate(v_values, start=2):
        if isinstance(value, matlab.double):
            value = value[0]  # Handle MATLAB double arrays
        
        elif isinstance(value, str) and value.startswith('matlab.double'):
            numeric_part = re.findall(r'\d+\.?\d*', value)[0]
            value = float(numeric_part)
        
        try:
            cell_value = float(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert value to float: {value} (Type: {type(value)})") from e
        
        cell = ws.cell(row=idx, column=insert_col, value=cell_value)
        cell.number_format = sci_format

    wb.save(new_path)
    wb.close()

    # Return the dataframe and file path
    df = pd.read_excel(new_path, sheet_name='Reaction List')
    print(f"New file saved to: {new_path}")
    return df, new_path

def perform_optknock(reaction_list, eng, df_v_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense):
    """
    Performs the OptKnock optimization for a given set of reactions.
    """
    try:
        selected_rxn_str = "{" + ", ".join([f"'{rxn}'" for rxn in reaction_list]) + "}"
        rxn_list_str = "{" + ", ".join([f"'{rxn}'" for rxn in constr_opt_rxn_list]) + "}"
        eng.eval(f"selectedRxnList = {selected_rxn_str};", nargout=0)

        eng.eval(f"options.targetRxn='{target_rxn}'", nargout=0)
        eng.eval(f"options.vmax={vmax}", nargout=0)
        eng.eval(f"options.numDel={num_del}", nargout=0)
        eng.eval(f"options.numDelSense='{num_del_sense}'", nargout=0)
        eng.eval(f"constrOpt.rxnList={rxn_list_str}", nargout=0)
        eng.eval(f"constrOpt.values=[{constr_opt_values1}*FBAsolution.f,{constr_opt_values2}]", nargout=0)
        eng.eval(f"constrOpt.sense='{constr_opt_sense}'", nargout=0)
        eng.eval(f"OptKnockSol=OptKnock(model,selectedRxnList,options,constrOpt)", nargout=0)

        # Retrieve OptKnock results
        OptKnockSol = eng.workspace['OptKnockSol']
        return OptKnockSol

    except Exception as e:
        print(f"Error during OptKnock computation: {e}")
        raise


def add_optknock_results_to_excel(reaction_list, eng, wb, OptKnockSol,new_col_name):
    """
    Adds the results from the OptKnock optimization to the Excel workbook.
    """
    rxnList = eng.eval("OptKnockSol.rxnList", nargout=1)
    fluxes_values = eng.eval("OptKnockSol.fluxes", nargout=1)
    

    print("rxnList:", rxnList)
   

    # Update reaction list by removing optimized reactions
    for rxn in OptKnockSol['rxnList']:
        if rxn in reaction_list:
            reaction_list.remove(rxn)

    ws = wb['Reaction List']
    insert_col = 1
    while insert_col <= ws.max_column:
        cell = ws.cell(row=1, column=insert_col)
        if cell.value is None:
            break
        insert_col += 1
    else:
        insert_col = ws.max_column + 1  # Insert at the end if no empty cells

    ws.cell(row=1, column=insert_col, value=new_col_name)

    # Write flux values to the new column
    sci_format = '0.#####E+00'
    for idx, value in enumerate(fluxes_values, start=2):
        if isinstance(value, matlab.double):
            value = value[0]  # Handle MATLAB double arrays
        
        elif isinstance(value, str) and value.startswith('matlab.double'):
            numeric_part = re.findall(r'\d+\.?\d*', value)[0]
            value = float(numeric_part)
        
        try:
            cell_value = float(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert value to float: {value} (Type: {type(value)})") from e
        
        cell = ws.cell(row=idx, column=insert_col, value=cell_value)
        cell.number_format = sci_format

    return wb  

def add_fluxes(xlsx_filename, rxnList, fluxes):
    """
    Adds fluxes
    """
    assert xlsx_filename.endswith('.xlsx')
    pass


def check_termination_condition(wb, count, consecutive_failures,eng):
    try:
        ws = wb['Reaction List']
    except KeyError:
        print("错误：未找到 'Reaction List' 工作表")
        return True  # 终止循环
    
    # 查找 v_values 列
    v_col = None
    for col in range(1, ws.max_column + 1):
        if ws.cell(1, col).value == 'v_values':
            v_col = col
            break
    if v_col is None:
        print("错误：未找到 'v_values' 列")
        return True  # 终止循环
    
    last_row = ws.max_row
    if last_row < 2:  # 至少需要1行数据
        return False
    
    # 获取最后一列和 v_values 列的值
    rxnList = eng.eval("OptKnockSol.rxnList", nargout=1)
   
    last_col_name = str(rxnList)
    if last_col_name.startswith("['") and last_col_name.endswith("']"):
            last_col_name = last_col_name[2:-2]
    last_col = None
    for col in range(1, ws.max_column + 1):
        if ws.cell(1, col).value == last_col_name:
            last_col = col
            break
    if last_col is None:
        print("错误：未找到 'optknock' 列")
        return True  # 终止循环

    last_val = ws.cell(last_row, last_col).value
    v_val = ws.cell(last_row, v_col).value
    # last_val = float(last_val)
    print('!!!!!!!!!!')
    print(v_val)
    print(last_val)
    # 验证数值类型
    if not isinstance(last_val, (int, float)) or not isinstance(v_val, (int, float)):
        print("警告：最后一列或 v_values 列包含非数值数据")
        return False
    
    # 更新连续失败计数器
    if last_val < v_val:
        consecutive_failures[0] += 1
    else:
        consecutive_failures[0] = 0  # 重置计数器
    
    # 终止条件：连续两次失败 或 超过20次循环
    if consecutive_failures[0] >= 2 or count >= 20:
        if count >= 20:
            print('未找到解')



        if consecutive_failures[0] >= 2:
            # 删除最后两列
            for col in range(last_col - 1, last_col + 1):
                for row in range(1, last_row + 1):
                    ws.cell(row=row, column=col, value=None)
        return True
    return False

def Handling_Currency_Metabolites(pairs_of_currency_metabolites_file,special_currency_metabolites_file):
    pairs_of_currency_metabolites = pd.read_csv(pairs_of_currency_metabolites_file)
    special_currency_metabolites = pd.read_csv(special_currency_metabolites_file)

    
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
    return pi_pairs1,pi_pairs2,h_pairs1,h_pairs2,nh4_pairs,other_pairs,currency_mets



def get_metpair(rea, pi_pairs1, h_pairs1, pi_pairs2, h_pairs2, nh4_pairs, other_pairs, currency_mets,phmet_list):
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
    phmet += phmet_list
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

def draw_plotly_graph(G,path):
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
        print(f'图片已保存到{path}')



# Example usage
if __name__ == "__main__":
    target_rxn = 'r_4693'
    vmax = 1000 
    num_del = 1
    num_del_sense = 'L'
    constr_opt_rxn_list = {'r_2111', 'r_4046'}
    constr_opt_values1 = 0.5
    constr_opt_values2 = 0.7
    constr_opt_sense = 'GE' 
    model_path = 'data/test1.xlsx'
    model_path_mat="data/Yeast8-OA.mat"
    reaction_list_path = 'data/selectedRxnList-172.xlsx'
    flux_value_file = 'data/test1v_valuesoptknock.xlsx'
    phmet_list=['s_0529[c]', 's_0530[w]', 's_0531[a]', 's_0532[m]', 's_0533[n]', 's_0534[p]', 's_2785[x]', 's_2857[q]', 's_3321[h]']
    pairs_of_currency_metabolites_file = "data/pairs_of_currency_metabolites.csv"
    special_currency_metabolites_file = "data/special_currency_metabolites.csv"

    full_pipeline(model_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense,phmet_list,pairs_of_currency_metabolites_file,special_currency_metabolites_file,model_path_mat,flux_value_file)