import matlab.engine
import pandas as pd
import shutil
import os
from openpyxl import load_workbook
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers

def full_pipeline(model_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense):
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
    while True:
        count += 1
        OptKnockSol = perform_optknock(reaction_list, eng, df_v_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense)
        
        # Write results to the workbook
        wb = add_optknock_results_to_excel(reaction_list, eng, wb, OptKnockSol)
        
        # Save the updated workbook
        wb.save(new_path)
        print(f"Saved results after {count} iterations")  # Print iteration count

        # Check termination condition (e.g., stop after 9 iterations)
        if check_termination_condition(wb,count,consecutive_failures,eng):

            break

    # Part 1 ends, part 2 begins (could include more operations like model editing, generating reports, etc.)
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


def add_optknock_results_to_excel(reaction_list, eng, wb, OptKnockSol):
    """
    Adds the results from the OptKnock optimization to the Excel workbook.
    """
    rxnList = eng.eval("OptKnockSol.rxnList", nargout=1)
    fluxes_values = eng.eval("OptKnockSol.fluxes", nargout=1)
    new_col_name = str(rxnList)
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


# TODO: Dummy function for checking termination condition
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
    reaction_list_path = 'data/selectedRxnList-172.xlsx'

    full_pipeline(model_path, reaction_list_path, target_rxn, vmax, num_del, num_del_sense, constr_opt_rxn_list, constr_opt_values1, constr_opt_values2, constr_opt_sense)