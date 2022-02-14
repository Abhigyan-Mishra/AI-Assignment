import numpy as np
import matplotlib.pyplot as plt
from utils.qubo import *
from utils.utility import *
from utils.qga import *
import time
import math
import pandas as pd


def q_f(data, qubo):
    qubo = qubo.values[:,1:(qubo.shape[0]+1)]
    n = qubo.shape[1]     # number of qubits and quantum systems
    q = (1/np.sqrt(2))*np.ones((2,n))  # Initialization of Quantum system with equal superposition
    print('We have {} number of Quantum systems'.format(n))
    current_systems = n*[q]
    iteration,T = 1,100
    iteration_count = []
    Fits, temps = [], []

    migration_count,replacement_count, mutation_count = 0,0,0
    beta_squared, detail, fitness_detail = [], [], []
    replacement_records, mutation_records, migration_records = [], [], []

    sys_prob_log = np.zeros(n)    # systems's probability log report
    all_sys_prob_log = []

    start_time = time.time()
    while sys_prob_log.min() < 99.5 and T>1:
        iteration_count.append(iteration)
        fit_current = [Measure_Eval2(system,qubo) for system in current_systems]
        rotated_systems = [Rotate(T,system) for system in current_systems]
        fit_rotated = [Measure_Eval2(system,qubo) for system in rotated_systems]    
        replaced_systems,fit_replaced,replacement_count,del_fit_rep,ran_rep,prob_rep = Replacement0(current_systems,rotated_systems,fit_current,fit_rotated,T,replacement_count)
        replacement_records.append(replacement_count)   
        # fit_replaced = [Measure_Eval3(system,qubo) for system in replaced_systems] # not required

        mutated_systems = [NOT_gate(system) for system in replaced_systems]
        fit_mutated = [Measure_Eval2(system,qubo) for system in mutated_systems]    
        adapted_systems,fit_adapted,mutation_count,del_fit_rep,ran_rep,prob_rep = Mutation0(replaced_systems,mutated_systems,fit_replaced,fit_mutated,T,mutation_count)
        mutation_records.append(mutation_count)
            

        migrated_systems,fit_migrated,migration_count,diff_fit_mig,ran_mig,prob_mig = Migration0(adapted_systems,fit_adapted,T,migration_count)
        migration_records.append(migration_count)

        sys_prob_log = np.array([Final_Measure(system)[1] for system in migrated_systems])
        all_sys_prob_log.append(sys_prob_log)

        fitness_detail.append(fit_current)
        fitness_detail.append(fit_rotated)
        fitness_detail.append(fit_replaced)
        fitness_detail.append(fit_mutated)
        fitness_detail.append(fit_migrated)
        fitness_detail.append([iteration,'$','$','$','$'])

        #################################################################
        Fits.append(fit_migrated)
        current_systems = migrated_systems
        beta_squared.append([system[1,:]**2 for system in current_systems])
        ################################################################
        
        detail.append(del_fit_rep) 
        detail.append(ran_rep)
        detail.append(prob_rep)
        detail.append(diff_fit_mig)
        detail.append(ran_mig)
        detail.append(prob_mig)   
        detail.append([iteration,'$','$','$','$'])  


        print('\n\n\n')
        temps.append(T)
        T = T*0.99
        iteration += 1

    end_time = time.time()
    results = [Final_Measure(system) for system in current_systems]
    best_result,E = best_system(results,qubo)

        
    class_prob, acc = predictions(data)

    measures = {
        "class_probabilities": class_prob,
         "best_result": best_result,
         "E": round(E, 4), 
         "sum_best_result":
          sum(best_result), 
          "migration_count": migration_count,
          "replacement_count" :replacement_count, 
          "mutation_count": mutation_count , 
          "iteration" :iteration, 
          "fitness": fitness_detail, 
          "computation_time": end_time - start_time, 
          "replacement_count": replacement_count, 
          "migration_count": migration_count, 
          "accuracy": acc, 
          "mutation_count": mutation_count, 
          "results": results
    }
    
    return measures
