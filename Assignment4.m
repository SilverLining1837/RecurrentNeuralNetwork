% Synthesize Function %

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

% Reading The Text and Creating Key Maps %

book_fname = 'data/Goblet.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid, '%c');
fclose(fid);

book_chars = unique(book_data);

dimension = 81;

numbers = [1:dimension];

for i = 1 : dimension
    chars{i} = book_chars(i);
    numbers_cell{i} = numbers(i);
end    

char_to_ind = containers.Map(chars,numbers);
ind_to_char = containers.Map(numbers,chars);

% Forward Pass and Loss Calculation %

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

% MAIN %

ReadText();
rng(400);
m = 100;
eta = 0.1;
seq_length = 25;
K = 81;
sig = 0.01;
n_epochs = 5;
tracker = 1;
epsilon = 0.0000001;
n = 200;
smooth_loss = 0;
step_counter = 1;
h0 = zeros(m,1);
hprev = h0;
RNN.W = randn(m, m) * sig;
RNN.U = randn(m, K) * sig;
RNN.V = randn(K, m) * sig;
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);

ada.W = zeros(m, m) ;
ada.U = zeros(m, K) ;
ada.V = zeros(K, m) ;
ada.b = zeros(m, 1);
ada.c = zeros(K, 1);

arr_count = 1;


for epochs = 1 : n_epochs
    
    disp('-------Epoch--------');
    while(tracker <= length(book_data) - seq_length - 1)
        X_chars = book_data(tracker:tracker + seq_length - 1);
        Y_chars = book_data(tracker + 1:tracker + seq_length);

        X = zeros(K, seq_length);
        Y = zeros(K, seq_length);

        for i =1:seq_length
           X (char_to_ind(X_chars(i)), i ) = 1;
           Y (char_to_ind(Y_chars(i)), i ) = 1;
        end

        x0 = X(:,1);

        [loss, p, h, a] = ForwardPass(hprev, X, Y, seq_length, RNN.W, RNN.U, RNN.V,RNN.b, RNN.c, m, K);
        
        [grad.W, grad.U, grad.V, grad.b, grad.c] = ComputeGradients(X, Y, p, seq_length, RNN.W, RNN.V, h, a, m, K, hprev);

        for f2 = fieldnames(RNN)'
            ada.(f2{1}) = ada.(f2{1}) + grad.(f2{1}).^2;
            RNN.(f2{1}) = RNN.(f2{1}) - (  eta * grad.(f2{1})  ) ./ (ada.(f2{1}) + epsilon).^(1/2);
            
           
        end
         if(mod(step_counter,100) == 0)
                smooth_loss = 0.999*smooth_loss + 0.001*loss;
                loss_array{arr_count} = smooth_loss;
                arr_count = arr_count + 1;
         end
            
        hprev = h(:,seq_length);

        

        tracker = tracker + seq_length;
        step_counter = step_counter + 1;
       
    end
    
    
    
    hfinal = hprev;
    hprev = h0;
    step_counter = 1;
    tracker = 1;
end

try_Y = Synthesize (hfinal, x0, n, RNN.W, RNN.U, RNN.V,RNN.b, RNN.c, m, K);

for i = 1:n
    fprintf('%s', ind_to_char(find(try_Y(:,i))));
end 
    



