import matlab.engine

# 启动 MATLAB 引擎
eng = matlab.engine.start_matlab()
eng.eval("initCobraToolbox", nargout=0)
# 定义 Python 列表（有序）
constr_opt_rxn_list = ['r_2111', 'r_4046']

try:


    # 生成元胞数组字符串
    rxn_list_str = "{" + ", ".join([f"'{rxn}'" for rxn in constr_opt_rxn_list]) + "}"
    print(f"生成的 MATLAB 元胞数组字符串: {rxn_list_str}")

    # 构造 MATLAB 命令
 
    eng.eval(f"constrOpt.rxnList={rxn_list_str}", nargout=0)
 

except Exception as e:
    print(f"发生错误: {e}")
finally:
    eng.quit()