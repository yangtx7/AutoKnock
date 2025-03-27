import matlab.engine
import pandas as pd
import shutil
import os
from openpyxl import load_workbook
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers





def full_pipeline(model_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense):
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()

    # 运行 MATLAB 表达式
    eng.eval("initCobraToolbox", nargout=0)
    print("调用 MATLAB 版本号: ", eng.version())  # 获取 MATLAB 版本号

   # 读取文件
    model= read_model(model_path,eng)
    
 
    v_values = FBA_compute(eng)
    
    df_v,df_v_path=add_v_to_excel_newfile(model_path, 
                          v_values, 
                          new_column_name='v_values',
                          suffix='v_values')


    # matlab_reaction_list=transfer_reaction_list(excel_file_path=reaction_list_path)
    
    rxnList=[]

    while True:
        OptKnockSol=perform_optknock(rxnList,eng,df_v_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense)
        print("--- 数据结构分析 ---")
        print(f"变量: {OptKnockSol['rxnList']}")





        add_optknock_results_to_excel(eng,df_v_path, OptKnockSol,suffix='optknock')
        
           
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
    
    insert_col = 1
    # 遍历第一行的所有列（直到找到第一个空单元格或末尾）
    while insert_col <= ws.max_column:
        cell = ws.cell(row=1, column=insert_col)
        if cell.value is None:
            break
        insert_col += 1
    else:
        insert_col = ws.max_column + 1  # 如果所有列都有内容，插入到末尾
    # 处理重复列名（保持原有逻辑）
    headers = [cell.value for cell in ws[1]]
    new_col_name = new_column_name
    counter = 1
    while new_col_name in headers:
        new_col_name = f"{new_column_name}_{counter}"
        counter += 1

    # 写入新列头到第一个空白列
    ws.cell(row=1, column=insert_col, value=new_col_name)
    
    # 写入数据并设置科学计数法格式
    sci_format = '0.#####E+00'
    for idx, value in enumerate(v_values, start=2):
        # 关键修改点：处理 MATLAB 数据类型
        if isinstance(value, matlab.double):
            # MATLAB 返回的数组，取第一个元素
            value = value[0]  # 如果是多维数组需要递归处理
            
        # 兼容性处理：如果意外以字符串形式传递
        elif isinstance(value, str) and value.startswith('matlab.double'):
            # 从字符串中提取数值（备用方案）
            numeric_part = re.findall(r'\d+\.?\d*', value)[0]
            value = float(numeric_part)
            
        # 强制类型转换确保安全
        try:
            cell_value = float(value)
        except ValueError as e:
            raise ValueError(f"无法转换值到浮点数: {value} (类型: {type(value)})") from e
            
        cell = ws.cell(row=idx, column=insert_col, value=cell_value)
        cell.number_format = sci_format
    
    # 保存并关闭
    wb.save(new_path)
    wb.close()
    
    # 返回结果
    df = pd.read_excel(new_path, sheet_name='Reaction List')
    print(f"新文件已保存至：{new_path}")
    return df,new_path



def perform_optknock(rxnList,eng, df_v_path,reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense):
    
    try:
        
        df = pd.read_excel(reaction_list_path)
        reaction_list = df.iloc[:, 0].tolist()  # 获取第一列数据

        filtered_rxn_list = [rxn for rxn in reaction_list if rxn not in rxnList]
    
       

            # 直接在 MATLAB 中创建元胞数组

        
        selected_rxn_str = "{" + ", ".join([f"'{rxn}'" for rxn in filtered_rxn_list]) + "}"



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

     

      

        # 从 MATLAB 工作区获取 OptKnockSol
        OptKnockSol = eng.workspace['OptKnockSol']
        return OptKnockSol

    except Exception as e:
        print(f"执行 OptKnock 计算时出现错误: {e}")
        raise  # 重新抛出异常以便调试


def add_optknock_results_to_excel(eng,df_v_path, OptKnockSol,suffix='optknock'):
    
            base = os.path.splitext(df_v_path)[0]
            ext = os.path.splitext(df_v_path)[1]
            new_path = f"{base}{suffix}{ext}"
            rxnList = eng.eval("OptKnockSol.rxnList", nargout=1)
            fluxes_values= eng.eval("OptKnockSol.fluxes", nargout=1)
            new_col_name=str(rxnList)
            print(new_col_name)
            
            # 复制原始文件
            if os.path.exists(new_path):
                os.remove(new_path)
            shutil.copyfile(df_v_path, new_path)
            
            # 操作副本文件
            wb = load_workbook(new_path)
            
            # 检查目标工作表
            if 'Reaction List' not in wb.sheetnames:
                wb.close()
                os.remove(new_path)
                raise ValueError("文件中没有'Reaction List'工作表")
            
            ws = wb['Reaction List']
            print(fluxes_values)
            
            insert_col = 1
            # 遍历第一行的所有列（直到找到第一个空单元格或末尾）
            while insert_col <= ws.max_column:
                cell = ws.cell(row=1, column=insert_col)
                if cell.value is None:
                    break
                insert_col += 1
            else:
                insert_col = ws.max_column + 1  # 如果所有列都有内容，插入到末尾
           

            # 写入新列头到第一个空白列
            ws.cell(row=1, column=insert_col, value=new_col_name)
            
            # 写入数据并设置科学计数法格式
            sci_format = '0.#####E+00'
            for idx, value in enumerate(fluxes_values, start=2):
                # 关键修改点：处理 MATLAB 数据类型
                if isinstance(value, matlab.double):
                    # MATLAB 返回的数组，取第一个元素
                    value = value[0]  # 如果是多维数组需要递归处理
                    
                # 兼容性处理：如果意外以字符串形式传递
                elif isinstance(value, str) and value.startswith('matlab.double'):
                    # 从字符串中提取数值（备用方案）
                    numeric_part = re.findall(r'\d+\.?\d*', value)[0]
                    value = float(numeric_part)
                    
                # 强制类型转换确保安全
                try:
                    cell_value = float(value)
                except ValueError as e:
                    raise ValueError(f"无法转换值到浮点数: {value} (类型: {type(value)})") from e
                    
                cell = ws.cell(row=idx, column=insert_col, value=cell_value)
                cell.number_format = sci_format
            
            # 保存并关闭
            wb.save(new_path)
            wb.close()
            
            # 返回结果
            df = pd.read_excel(new_path, sheet_name='Reaction List')
            print(f"新文件已保存至：{new_path}")
            return df,new_path


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
    model_path='C:/Users/Administrator/Desktop/model_input.xlsx'
    reaction_list_path='C:/Users/Administrator/Desktop/selectedRxnList-172.xlsx'

   
    full_pipeline(model_path, reaction_list_path,target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1,constr_opt_values2, constr_opt_sense)
    