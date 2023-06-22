function y = calcs(x,coeff)

if size(x,2)~=1, error('x should be a column'), end
if size(coeff,2)~=1, error('coeff should be a column'), end

if ~isempty(find(x==0)), error('x contains zeros!'), end

if length(coeff)==4
    y=exp([log(x).^3 log(x).^2 log(x) ones(length(x),1)]*coeff);
elseif length(coeff)==2
    y=exp([log(x) ones(length(x),1)]*coeff);
else 
    error('length of coeff is incorrect!\n');
end