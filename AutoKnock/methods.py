
import pandas as pd


import itertools
import networkx as nx
import plotly.graph_objects as go






def FBA_compute(eng):
    """
    Computes flux balance analysis (FBA) for the given model.
    """
    eng.eval("FBAsolution = optimizeCbModel(model)", nargout=0)  # Assume 'model' is defined previously
    # Extract v values (flux values)
    v_values = eng.eval("FBAsolution.v", nargout=1)
    print("FBA computation complete. Flux values extracted.")
    return v_values


def perform_optknock(reaction_list, eng, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense):
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