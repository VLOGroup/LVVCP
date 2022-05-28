import torch

def add_projection_scalar(scalar_param, init=1., min=0, max=1000, lr_mul=None, name_tag=None, mode=''):
    """ sets the value of the parameter to the initial vlaue from the config and
        adds a projection function to the parameter.
        Requires Block Adam optimizer to work!
        Right after the GD step of ADAM the .proj function will be called by BlockAdam
        This allows to limit the data to an allowed range
    """
    assert type(scalar_param) == torch.nn.parameter.Parameter, f"parameter must be a pytorch parameter"
    scalar_param.data = torch.tensor(init, dtype=scalar_param.dtype)
    # add a positivity constraint
    scalar_param.proj = lambda: scalar_param.data.clamp_(min, max)
    if lr_mul is not None:
        scalar_param._lr_mul = lr_mul
    if name_tag is not None:
        scalar_param._name_tag = name_tag
    if mode:
        assert mode == 'learned', f"if constraints are used, the mode must be set to learned!"

def add_parameter(obj, name, config):
    assert name in config, f"Parameter '{name}' not found in config"
    assert 'mode' in config[name], f"Parameter '{name}' does not have a mode setting: '{config[name]}'"
    assert not hasattr(obj, name), f"A parameter '{name}' is already present on ojb:'{obj}''"
    if config[name]['mode'] == 'fixed':  # Fixed Mode:=> generate a buffer that will be saved with model
        scalar_param = torch.tensor(config[name]['init'])
        obj.register_buffer(name, torch.tensor(config[name]['init']))
    elif config[name]['mode'] == 'learned':  # Learned Mode:=> generate a Parameter
        scalar_param = torch.tensor(1.0) # must be floating point type to get gradients
        scalar_param = torch.nn.Parameter(scalar_param)
        add_projection_scalar(scalar_param, name_tag=name,  **config[name])
        setattr(obj, name, scalar_param)
    else:
        raise RuntimeError(f"mode {config[name]['mode']} unknown! for parameter {name} and modeconfig {config[name]}")
    scalar_param._is_custom_scalar_param = True

def get_parameter_v2(name, config, **kwargs):
    """ """
    assert 'mode' in config, f"Parameter does not have a mode setting: '{config}'"
    if config['mode'] == 'learned':  # Learned Mode:=> generate a Parameter
        scalar_param = torch.tensor(config['init']) # must be floating point type to get gradients
        scalar_param = torch.nn.Parameter(scalar_param)
        add_projection_scalar(scalar_param, name_tag=name,  **config)
    elif config[name]['mode'] == 'fixed':  # Fixed Mode:=> generate a buffer that will be saved with model
        if 'parent' in kwargs:
            scalar_param = torch.tensor(config[name]['init'])
            kwargs['parent'].register_buffer(name, torch.tensor(config[name]['init']))
        else:
            raise ValueError(f"Setting a parameter as a fixed buffer requires access to the parent. Pass the calling object as parent=obj")
    else:
        raise RuntimeError(f"mode {config['mode']} unknown! for parameter {name} and modeconfig {config}")
    scalar_param._is_custom_scalar_param = True
    return scalar_param



def get_model_custom_scalar_params(model):
    return {name:val for name,val in model.named_parameters() if  hasattr(val, '_is_custom_scalar_param') }