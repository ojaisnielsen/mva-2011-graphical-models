function s = logsumexp(x)

x = sort(reshape(x, numel(x), 1), 'descend');
s = 1;
for i=2:numel(x)
    s = s + exp(x(i)-x(1));
end
s = log(s) + x(1);

end

