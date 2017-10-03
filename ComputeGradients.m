function [grad_W, grad_U, grad_V, grad_b, grad_c] = ComputeGradients(x, y, p, n, W, V, h, a, m, K, hprev)
    
    grad_W = zeros(m, m);
    grad_U = zeros(m, K);
    grad_V = zeros(K, m);
    grad_c = zeros(K, 1);
    grad_b = zeros(m, 1);
    

    o_d{n} = -( y(:,n)-p(:,n) )';
    h_d{n} = o_d{n} * V;
    a_d{n} = h_d{n} * (diag(1- tanh(a(:,n) ).^2 ));
    
    
    for i = (n-1) : -1 : 1
        o_d{i} = -( y(:,i)-p(:,i) )';
        h_d{i} = o_d{i} * V + a_d{i+1} * W;
        a_d{i} = h_d{i} * (diag(1- tanh(a(:,i) ).^2 ));
    end
    
    for last = 1 : n
        grad_V = grad_V + (o_d{last})' * h(:,last)' ;
        
        grad_U = grad_U + (a_d{last})' * x(:,last)';
        
        if last == 1
            grad_W = grad_W + (a_d{last})' * hprev' ;
            
        else
            grad_W = grad_W + (a_d{last})' * h(:,last-1)' ;
        end
        
        
        grad_b = grad_b + a_d{last}';
        grad_c = grad_c + o_d{last}';
           
   
       
    end  
end