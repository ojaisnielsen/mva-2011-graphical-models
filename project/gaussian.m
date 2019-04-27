function G = gaussian(sigma)

if sigma == 0,
    G = 1;
    return
end

r = round(3*sigma);

[x,y] = meshgrid(-r:r,-r:r);

G = exp(-(x.^2 + y.^2)/(2*sigma^2));

G = G / sum(sum(G));

end

