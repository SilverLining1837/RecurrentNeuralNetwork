function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
end

function grad = ComputeGradNum(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
     [l1, p, hwewe, a] = ForwardPass(hprev, X, Y, 25, RNN_try.W, RNN_try.U, RNN_try.V,RNN_try.b, RNN_try.c, 100, 81);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    [l2, p, hewewe, a] = ForwardPass(hprev, X, Y, 25, RNN_try.W, RNN_try.U, RNN_try.V,RNN_try.b, RNN_try.c, 100, 81);
    grad(i) = (l2-l1)/(2*h);
end

