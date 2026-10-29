import jax
import jax.numpy as jnp

### MODE ###
"""
    Calculates the most frquent element in axis=0 direction for 2d arrays (downward, columnwise)
    Returns the uppermost if there are more than one most occuring element
    Example :
    x = [[0, 1, 2, 2, 4],
         [0, 2, 1, 3, 3],
         [1, 1, 1, 1, 1],
         [2, 2, 1, 3, 0]]
         
    result = mode(x)
    
    => result = [0, 1, 1, 3, 4] 
"""

def mode_first_occurrence(arr):
    # Broadcast arr to compare each element with all elements
    # Shape: [len(arr), len(arr)]
    cmp = arr[:, None] == arr[None, :]  # boolean mask
    counts = jnp.sum(cmp, axis=1)       # how often each value appears

    max_count = jnp.max(counts)
    is_mode = counts == max_count

    # First occurrence of a mode
    first_mode_idx = jnp.argmax(is_mode)
    return arr[first_mode_idx]

# Vectorized across columns (axis=1)
def mode(x):
    return jax.vmap(mode_first_occurrence, in_axes=1)(x)
    

# Mode along any axis ove any dimensional array
def mode_along_axis(axis):
    def mode_along_axis_helper(x):
        def mode_fn(values):
            # values: shape (axis_size,)
            cmp = values[:, None] == values[None, :]  # shape (axis_size, axis_size)
            counts = jnp.sum(cmp, axis=1)
            max_count = jnp.max(counts)
            is_mode = counts == max_count
            return values[jnp.argmax(is_mode)]
    
        # Move the target axis to the front
        x_moved = jnp.moveaxis(x, axis, 0)  # shape: (axis_size, ...)
    
        # Collapse the rest of the dimensions into a single one
        flat_shape = (x_moved.shape[0], -1)
        x_flat = x_moved.reshape(flat_shape)
    
        # Apply mode function to each column (i.e., original subarray across `axis`)
        modes = jax.vmap(mode_fn, in_axes=1)(x_flat)
    
        # Reshape back to original shape without axis
        out_shape = x.shape[:axis] + x.shape[axis+1:]
        return modes.reshape(out_shape)
    
    return mode_along_axis_helper
