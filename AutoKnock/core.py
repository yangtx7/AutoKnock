import matlab.engine
import pandas as pd
import shutil
import os
from openpyxl import load_workbook
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers





def full_pipeline(model_path, reaction_list_path,selected_rxn_list_matlab, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values, constr_opt_sense):
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    # 运行 MATLAB 表达式
    eng.eval("initCobraToolbox", nargout=0)
    print("调用 MATLAB 版本号: ", eng.version())  # 获取 MATLAB 版本号

   # 读取文件
    model= read_model(model_path,eng)
    
 
    v_values = FBA_compute(eng)
    
    df_v=add_v_to_excel_newfile(model_path, 
                          v_values, 
                          new_column_name='v_values',
                          suffix='v_values')


    matlab_reaction_list=transfer_reaction_list(excel_file_path=reaction_list_path)
    
    

    while True:
        OptKnockSol=perform_optknock(model, matlab_reaction_list, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense)

        add_optknock_results_to_excel(OptKnockSol,df_v)
        
           
        if check_termination_condition():
            break

    # part 1结束，part 2开始

    # read_mat_file(mat_name2)

    # remove_CM()

    # edit_model()

    # draw_figures()

    # generate_report()

def read_model(model_path,eng):
    if model_path.endswith('.mat'):
        eng.eval(f"model = readcbModel('{model_path}')", nargout=0)
        print(f"已使用 readcbModel 读取 .mat 文件 {model_path}")
    elif model_path.endswith('.xlsx'):
        eng.eval(f"model = xls2model('{model_path}')", nargout=0)
        print(f"已使用 xls2model 读取 .xlsx 文件 {model_path}")
        # 读取 Excel 文件，指定 sheet_name 为 None 以读取所有表
        excel_file = pd.ExcelFile(model_path)
        
        return  excel_file
    else:
        print("不支持的文件类型，请提供 .mat 或 .xlsx 文件。")



def FBA_compute(eng):
    """
    Computes FBA
    """
   
    eng.eval("FBAsolution = optimizeCbModel(model)", nargout=0)  #此处的model是一个字符串，上述读取过程中已经默认命名模型为model
    # 提取 v 值
    v_values = eng.eval("FBAsolution.v", nargout=1)
   
    print("FBA 计算完成，已提取 v 值。")
    return v_values



def add_v_to_excel_newfile(excel_path, 
                          v_values, 
                          new_column_name='v_values',
                          suffix='_modified'):
    # 生成新文件路径
    base = os.path.splitext(excel_path)[0]
    ext = os.path.splitext(excel_path)[1]
    new_path = f"{base}{suffix}{ext}"
    
    # 复制原始文件
    if os.path.exists(new_path):
        os.remove(new_path)
    shutil.copyfile(excel_path, new_path)
    
    # 操作副本文件
    wb = load_workbook(new_path)
    
    # 检查目标工作表
    if 'Reaction List' not in wb.sheetnames:
        wb.close()
        os.remove(new_path)
        raise ValueError("文件中没有'Reaction List'工作表")
    
    ws = wb['Reaction List']
    
    # 验证数据长度
    data_row_count = ws.max_row - 1  # 排除标题行
    if len(v_values) != data_row_count:
        wb.close()
        os.remove(new_path)
        raise ValueError(f"数据行数不匹配（v值:{len(v_values)}，工作表:{data_row_count}）")
    
    # 处理重复列名
    headers = [cell.value for cell in ws[1]]
    new_col_name = new_column_name
    counter = 1
    while new_col_name in headers:
        new_col_name = f"{new_column_name}_{counter}"
        counter += 1
    
    # 定位最后一列
    last_col = ws.max_column
    
    # 写入新列头
    ws.cell(row=1, column=last_col+1, value=new_col_name)
    
    # 写入数据并设置科学计数法格式
    sci_format = '0.00E+00'
    for idx, value in enumerate(v_values, start=2):
        # 处理MATLAB字符串
        if isinstance(value, str) and value.startswith('matlab.double'):
            numeric_part = re.findall(r'\d+\.?\d*', value)[0]
            value = float(numeric_part)
        cell = ws.cell(row=idx, column=last_col+1, value=float(value))
        cell.number_format = sci_format
    
    # 保存并关闭
    wb.save(new_path)
    wb.close()
    
    # 返回结果
    df = pd.read_excel(new_path, sheet_name='Reaction List')
    print(f"新文件已保存至：{new_path}")
    return  df

# def add_v_values_to_excel(model_path,model, v_values):
#     try:
#         # 获取所有表名
#         sheet_names = model.sheet_names
#         # 获取 Reaction List 工作表的数据
#         reaction_list_df = model.parse('Reaction List')

#         # 将 v 值转换为合适的格式（假设 v_values 是一维数组）
#         v_values_list = [val[0] for val in v_values]

#         # 确保 v 值的长度与 Reaction List 表的行数一致
#         if len(v_values_list) != len(reaction_list_df):
#             raise ValueError("v 值的长度与 Reaction List 表的行数不一致。")

#         # 添加 v 值作为新的一列
#         reaction_list_df['v_values'] = v_values_list

#         # 创建一个 Pandas ExcelWriter 对象，用于写入修改后的数据到 Excel 文件
#         with pd.ExcelWriter(model_path, engine='openpyxl') as writer:
#             # 遍历每个表名
#             for sheet_name in sheet_names:
#                 if sheet_name == 'Reaction List':
#                     # 如果是 Reaction List 表，则写入修改后的数据
#                     reaction_list_df.to_excel(writer, sheet_name=sheet_name, index=False)
#                 else:
#                     # 对于其他表，直接写入原始数据
#                     df = model_path.parse(sheet_name)
#                     df.to_excel(writer, sheet_name=sheet_name, index=False)
#                     return df

#         print(f"已将 v 值添加到 {model_path} 的 Reaction List 表的最后一列。")
#     except Exception as e:
#         print(f"处理 Excel 文件时出现错误: {e}")






def transfer_reaction_list(excel_file_path):
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file_path)
        # 假设 Excel 文件中只有一列，获取该列的数据
        reaction_list = df.iloc[:, 0].tolist()

        # 将 Python 列表转换为 MATLAB 元胞数组
        matlab_reaction_list = matlab.cell([1, len(reaction_list)])
        for i, reaction in enumerate(reaction_list):
            matlab_reaction_list[0, i] = matlab.double([reaction])

        # 在 MATLAB 工作区中创建 selectedRxnList 变量
        print("已将 Excel 文件中的待选反应集导入到 MATLAB 工作区的 selectedRxnList 中。")
        return matlab_reaction_list
        
    except Exception as e:
        print(f"处理 Excel 文件时出现错误: {e}")




def perform_optknock(model, selected_rxn_list_matlab, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense):
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    try:


        # 处理 constr_opt_rxn_list 为 MATLAB 元胞数组
        constr_opt_rxn_list_matlab = eng.cell(1, len(constr_opt_rxn_list))
        for i, rxn in enumerate(constr_opt_rxn_list):
            constr_opt_rxn_list_matlab[0, i] = rxn

        # 进行 FBA 计算
        fbaWT = eng.optimizeCbModel(model)
        constr_opt_values=[constr_opt_values1*fbaWT.f,constr_opt_values2]
        constr_opt_values_mat = matlab.double(constr_opt_values)





        # 设置 options 结构体
        options = eng.struct('targetRxn', target_rxn, 'vmax', vmax, 'numDel', num_del, 'numDelSense', num_del_sense)

        # 设置 constrOpt 结构体
        constrOpt = eng.struct('rxnList', constr_opt_rxn_list_matlab, 'values', constr_opt_values_mat, 'sense', constr_opt_sense)

        # 执行 OptKnock 计算
        OptKnockSol = eng.OptKnock(model, selected_rxn_list_matlab, options, constrOpt)

        return OptKnockSol

    except Exception as e:
        print(f"执行 OptKnock 计算时出现错误: {e}")


def add_optknock_results_to_excel(OptKnockSol, df_v):
    try:
        # 提取 rxnList 和 fluxes
        rxn_list = [OptKnockSol['rxnList'][0, i][0] for i in range(len(OptKnockSol['rxnList'][0]))]
        fluxes = [OptKnockSol['fluxes'][i][0] for i in range(len(OptKnockSol['fluxes']))]

        # 创建一个 DataFrame 来存储结果
        result_df = pd.DataFrame({rxn_list[i]: [fluxes[i]] for i in range(len(rxn_list))})

        # 读取 Excel 文件
        excel_file = pd.ExcelFile(excel_file_path)
        reaction_list_df = excel_file.parse('Reaction List')

        # 将结果添加到 Reaction List 表的最后一列
        for col in result_df.columns:
            reaction_list_df[col] = result_df[col].values

        # 保存修改后的 DataFrame 到 Excel 文件
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            reaction_list_df.to_excel(writer, sheet_name='Reaction List', index=False)

        print(f"已将 OptKnock 结果添加到 {excel_file_path} 的 Reaction List 表的最后一列。")
    except Exception as e:
        print(f"处理 Excel 文件时出现错误: {e}")


def add_fluxes(xlsx_filename, rxnList, fluxes):
    """
    Adds fluxes
    """
    assert xlsx_filename.endswith('.xlsx')
    pass

def check_termination_condition():
    """
    Checks termination condition
    """
    return False


# 示例使用
if __name__ == "__main__":
   
    target_rxn='r_4693'
    vmax=1000 
    num_del=1
    num_del_sense='L'
    constr_opt_rxn_list={'r_2111','r_4046'}
    constr_opt_values1=0.5
    constr_opt_values2=0.7
    constr_opt_sense='GE' 
    model_path='C:/Users/Administrator/Desktop/副本Yeast8-OA.xlsx'
    reaction_list_path='C:/Users/Administrator/Desktop/selectedRxnList-172.xlsx'

   
    full_pipeline(model_path, reaction_list_path,target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense)
    