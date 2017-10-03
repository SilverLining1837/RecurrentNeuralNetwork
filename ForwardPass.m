function [loss, p, h, a] = ForwardPass(h0, x, y, n, W, U, V, b, c, m, K)
    
    p = zeros(K, n);
    h = zeros(m, n);
    a = zeros(m, n);
    o = zeros(K, 1);
    for i = 1 : n
        
        if(i == 1)
            a(:,i) = W * h0 + U * x(:,1) + b;
            h(:,i) = tanh(a(:,i));
            o(:,i) = V * h(:,i) + c;
            p(:,i) = softmax(o(:,i));
        else
            a(:,i) = W * h(:,i-1) + U * x(:,i) + b;
            h(:,i) = tanh(a(:,i));
            o(:,i) = V * h(:,i) + c;
            p(:,i) = softmax(o(:,i));
        end
    end
    
    sum = 0;
    for l = 1 : n
        sum = sum + log( (y(:,i))' * p(:,i) );
    end
        
    loss = -1 * sum;
    
    
end