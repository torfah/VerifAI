from verifai.features.features import *
from verifai.falsifier import mtl_falsifier
from dotmap import DotMap
from config import *
#import pandas
#pandas.set_option("display.max_rows", None, "display.max_columns", None)
init_conditions = Struct({
    'ego_target_speed': Box([65.0, 80.0]),
    'other_target_speed': Box([25.0, 30.0]),
    'initial_dist': Box([30.0, 40.0]),
    'clear_dist': Box([10.0, 15.0])
})

sample_space = {'init_conditions': init_conditions}

SAMPLERTYPE = 'ce'
MAX_ITERS = 2
MAXREQS = 5

specification = ['~(F[0, 10] dtc)']

falsifier_params = DotMap()
falsifier_params.n_iters = MAX_ITERS
falsifier_params.compute_error_table = True
falsifier_params.fal_thres = FALSE_THRESHOLD 
falsifier_params.n_sim_steps = N_SIM_STEP
server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

sampler_params = DotMap()
sampler_params.init_num = 2
falsifier_params.sampler_params = sampler_params

falsifier = mtl_falsifier(sample_space=sample_space, sampler_type=SAMPLERTYPE,
                          specification=specification, falsifier_params=falsifier_params,
                          server_options=server_options)
falsifier.run_falsifier()

analysis_params = DotMap()
analysis_params.k_closest_params.k = 4
analysis_params.random_params.count = 4
falsifier.analyze_error_table(analysis_params=analysis_params)

print("Falsified Samples")
print(falsifier.error_table.table)

print("Safe Samples")
print(falsifier.safe_table.table)
