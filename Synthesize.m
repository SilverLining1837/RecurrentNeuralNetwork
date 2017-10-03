function [Y] = Synthesize(h0, x0, n, W, U, V, b, c,m,K)
    Y = zeros(81,n);
    
    p = zeros(K, n);
    h = zeros(m, n);
    a = zeros(m, n);
    o = zeros(K, 1);
    
    for i = 1 : n
        
        if(i == 1)
            a(:,i) = W * h0 + U * x0 + b;
            h(:,i) = tanh(a(:,i));
            o(:,i) = V * h(:,i) + c;
            p(:,i) = softmax(o(:,i));
        else
            a(:,i) = W * h(:,i-1) + U * x + b;
            h(:,i) = tanh(a(:,i));
            o(:,i) = V * h(:,i) + c;
            p(:,i) = softmax(o(:,i));
        end
        cp = cumsum(p(:,i));
        ra = rand;
        ixs = find(cp-ra > 0);
        ii = ixs(1);
        Y(ii,i) = 1;
        x = Y(:,i);
    end
end