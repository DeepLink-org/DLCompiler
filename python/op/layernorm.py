import triton
import triton.language as tl

@triton.jit                                                                       
def _layer_norm_fwd_fused_with_small_n(                                                        
    X,  # pointer to the input                                                    
    Y,  # pointer to the output                                                   
    W,  # pointer to the weights                                                  
    B,  # pointer to the biases                                                   
    Mean,  # pointer to the mean                                                  
    Rstd,  # pointer to the 1/std                                                 
    stride,  # how much to increase the pointer when moving by 1 row              
    N,  # number of columns in X                                                  
    eps,  # epsilon to avoid division by zero                                     
    BLOCK_SIZE: tl.constexpr,                                                     
):                                                                                
    # Map the program id to the row of X and Y it should compute.                 
    row = tl.program_id(0)                                                        
    Y += row * stride                                                             
    X += row * stride                                                             
    # Compute mean and variance                                                   
    mean = 0                                                                      
    cols = tl.arange(0, BLOCK_SIZE)                                               
    x = tl.load(X + cols).to(tl.float32)                                          
    mean = tl.sum(x, axis=0) / N                                                  
    x_square = x * x                                                              
    var = tl.sum(x_square, axis=0) / N - mean * mean                              
    rstd = 1 / tl.sqrt(var + eps)                                                 
                                                                                                                                                                                                                                               
    # Write mean / rstd                                                           
    tl.store(Mean + row, mean)                                                    
    tl.store(Rstd + row, rstd)                                                    
                                                                                    
    # Normalize and apply linear transformation                                   
    w = tl.load(W + cols)                                                         
    b = tl.load(B + cols)                                                         
    x_hat = (x - mean) * rstd                                                     
    y = x_hat * w + b                                                             
    # Write output                                                                
    tl.store(Y + cols, y)