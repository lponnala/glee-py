% FDR: Procedure for multiple comparisons as descibed in Wichert et al
% (Bioinformatics, 2004) or Ahdesmaaki et al (BMC Bioinformatics, 2005)
% USAGE: H=fdr(P,q)
% P = column-vector of p-values from the individual tests
% q = expected proportion of false positives
% H = column-vector containing 1 where FDR rejects the null
% hypothesis, 0 at remaining indices
% NOTE: 
% 1. No need to worry about NaNs in the p-values, sorting P in ascending order will
% put them at the very end. This is not an issue unless FDR rejects in all
% but the last one case, only because it didn't have a valid p-value.
% 2. H will retain the same ordering as P, i.e. H(i) and P(i) will refer
% to the same signal
function H=fdr(P,q)

M=length(P);
[Ps,I]=sort(P); Is=q*(1:1:M)'/M; i_q=max(find(Ps<=Is));
H=zeros(M,1); H(I(1:i_q))=1;
