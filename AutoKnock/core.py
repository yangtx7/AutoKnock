def full_pipeline(mat_name1, xls_name, mat_name2):
    convert_mat_to_xlsx(mat_filename=mat_name1, xlsx_filename=xls_name)

    edit_xlsx(xlsx_filename=xls_name)

    convert_xlsx_to_mat(xlsx_filename=xls_name, mat_filename=mat_name2) 
 
    result = FBA_compute(mat_filename=mat_name2)
    
    add_FBA_v_value(result=result, xlsx_filename=xls_name)

    while True:
        rxnList, fluxes = optKnock_compute(xls_name)
        add_fluxes(xlsx_filename=xls_name, rxnList, fluxes)

        if check_termination_condition():
            break

    # part 1结束，part 2开始

    read_mat_file(mat_name2)

    remove_CM()

    edit_model()

    draw_figures()

    generate_report()



def convert_mat_to_xlsx(mat_filename, xlsx_filename):
    """
    Converts .mat files to .xlsx files
    """
    assert mat_filename.endswith('.mat')
    assert xlsx_filename.endswith('.xlsx')
    pass

def edit_xlsx(xlsx_filename):
    """
    Edits .xlsx files
    """
    assert xlsx_filename.endswith('.xlsx')
    pass

def convert_xlsx_to_mat(xlsx_filename, mat_filename):
    """
    Converts .xlsx files to .mat files
    """
    assert xlsx_filename.endswith('.xlsx')
    assert mat_filename.endswith('.mat')
    pass

def FBA_compute(mat_filename):
    """
    Computes FBA
    """
    assert mat_filename.endswith('.mat')
    result = None
    return result

def add_FBA_v_value(result, xlsx_filename):
    """
    Adds FBA v value
    """
    assert xlsx_filename.endswith('.xlsx')
    pass

def optKnock_compute():

    # read selectedRxnList from xlsx file and convert to matlab list

    # setting parameters

    # run optKnock

    # get rxnlist

    # recore fluxes

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