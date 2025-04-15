import argparse

def init_parametes():
    parser = argparse.ArgumentParser()

    

    parser.add_argument("--target_rxn", type=str, default='r_4693', help="Select the target reaction")
    parser.add_argument("--vmax", type=int, default=1000, help="Set the maximum velocity")
    parser.add_argument("--num_del", type=int, default=1, help="Set the number of deletions")
    parser.add_argument("--num_del_sense", type=str, default='L', help="Set the deletion sense")
    parser.add_argument("--constr_opt_rxn_list", type=lambda s: set(s.split(',')), default={'r_2111', 'r_4046'}, help="Set the constraint optimization reaction list, use comma to separate values")
    parser.add_argument("--constr_opt_values1", type=float, default=0.5, help="Set the first constraint optimization value")
    parser.add_argument("--constr_opt_values2", type=float, default=0.7, help="Set the second constraint optimization value")
    parser.add_argument("--constr_opt_sense", type=str, default='GE', help="Set the constraint optimization sense")
    parser.add_argument("--model_path", type=str, default='data/test1.xlsx', help="Set the path to the model file")
    parser.add_argument("--model_path_mat", type=str, default="data/Yeast8-OA.mat", help="Set the path to the MAT model file")
    parser.add_argument("--reaction_list_path", type=str, default='data/selectedRxnList-172.xlsx', help="Set the path to the reaction list file")
    parser.add_argument("--flux_value_file", type=str, default='data/test1v_valuesoptknock.xlsx', help="Set the path to the flux value file")
    parser.add_argument("--phmet_list", type=lambda s: s.split(','), default=['s_0529[c]', 's_0530[w]', 's_0531[a]', 's_0532[m]', 's_0533[n]', 's_0534[p]', 's_2785[x]', 's_2857[q]', 's_3321[h]'], help="Set the phmet list, use comma to separate values")
    parser.add_argument("--pairs_of_currency_metabolites_file", type=str, default="data/pairs_of_currency_metabolites.csv", help="Set the path to the pairs of currency metabolites file")
    parser.add_argument("--special_currency_metabolites_file", type=str, default="data/special_currency_metabolites.csv", help="Set the path to the special currency metabolites file")

    
    

    return parser
