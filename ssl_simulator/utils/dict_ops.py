"""
"""

__all__ = [
    "parse_kwargs",
]

#######################################################################################

def parse_kwargs(kwargs_input, kwargs_default):
    """
    Merge user-provided keyword arguments with default values.

    Parameters
    ----------
    kwargs_input : dict  
        Dictionary containing user-specified keyword arguments.  
    kwargs_default : dict  
        Dictionary containing default keyword arguments.  

    Returns
    -------
    dict  
        A dictionary where user-specified values override the defaults, while 
        preserving unspecified default values.  
    """
    kwargs = kwargs_default.copy()  # Avoid modifying the original default dictionary
    kwargs.update({k: v for k, v in kwargs_input.items() if k in kwargs_default})
    return kwargs

#######################################################################################