import pybamm 
import numpy as np

def simulate_spm(param_values: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the SPM model for given parameter values.
    :param param_values: Dictionary of parameter names to override (e.g., {"Ambient temperature [K]": 300.0, ...})
    :return: (time, voltage) as numpy arrays.
    """
    # Create SPM model and set geometry
    model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry

    # Load default parameter values and update with given values
    params = model.default_parameter_values
    params.update(param_values)
    params.process_model(model)
    params.process_geometry(geometry)

    # Create mesh and discretize model
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # Solve the model for a 1-hour discharge
    solver = model.default_solver
    t_end = 3600.0  # 1 hour in seconds
    n_pts = 100
    t_eval = np.linspace(0, t_end, n_pts)
    # Solve model (the solver may stop early if it hits voltage cutoff)
    solution = solver.solve(model, t_eval)
    time = solution.t  # time in seconds
    voltage = solution["Terminal voltage [V]"].entries  # voltage time series
    # If the solver terminated early (time shorter than t_end), pad voltage with last value to full length for consistency
    if time[-1] < t_end:
        # Create full time array to exactly match requested t_eval for output
        full_time = t_eval
        # Interpolate or extend voltage to full time range
        voltage_full = np.interp(full_time, time, voltage, right=voltage[-1])
        time = full_time
        voltage = voltage_full
    return time, voltage
