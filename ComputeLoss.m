function loss = ComputeLoss(X, Y, RNN_try, hprev)
P = ForwardPass(X,W1, W2, b1, b2);
for i = 1 : size(P,2)
    sum = sum + -log(Y(:,i)' * P(:,i));
end
end