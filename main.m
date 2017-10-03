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
            
%         
%         
        hprev = h(:,seq_length);

        

        tracker = tracker + seq_length;
        step_counter = step_counter + 1;
       
    end
    
%      if (mod(step_counter,1000) == 0)
    %        smooth_loss
%         end
%         if (mod(step_counter,1000) == 0)
            
%         end
    
    
    hfinal = hprev;
    hprev = h0;
    step_counter = 1;
    tracker = 1;
end

            try_Y = Synthesize (hfinal, x0, 1000, RNN.W, RNN.U, RNN.V,RNN.b, RNN.c, m, K);

            for i = 1:1000
                fprintf('%s', ind_to_char(find(try_Y(:,i))));
            end 
    



