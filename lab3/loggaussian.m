function y = loggaussian(x, mu, sigma)

[d, N] = size(x);
[~, K] = size(mu);
y = zeros(K,N);

for k=1:K
    for n=1:N
        y(k,n) = -0.5 * (d*log(2*pi) + log(det(sigma(:,:,k))) + ((x(:,n)-mu(:,k))' * inv(sigma(:,:,k)) * (x(:,n)-mu(:,k))));
    end
end

end

