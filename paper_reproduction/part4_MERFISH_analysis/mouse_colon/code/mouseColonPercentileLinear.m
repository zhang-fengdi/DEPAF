function result = mouseColonPercentileLinear(values, percentiles)
% NumPy percentile(method='linear'), including its stable interpolation order.
% Unlike prctile's default, rank = (n-1)*(p/100), using zero-based ranks.
validateattributes(values, {'numeric'}, {'vector','real','finite','nonempty'});
validateattributes(percentiles, {'numeric'}, {'vector','real','finite','>=',0,'<=',100});
x = sort(double(values(:)));
p = double(percentiles(:));
h = (numel(x)-1) .* (p./100);
lo = floor(h);
hi = min(lo+1, numel(x)-1);
g = h-lo;
a = x(lo+1); b = x(hi+1); difference = b-a;
y = a + difference.*g;
upper = g >= 0.5;
y(upper) = b(upper) - difference(upper).*(1-g(upper));
result = reshape(y, size(percentiles));
end
