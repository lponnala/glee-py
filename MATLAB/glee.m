clear;

warning off MCR:ClassInUseAtExit
warning off MATLAB:ClassInstanceExists

% % ---- USER-DEFINED ----
% NAME='Tip_vs_Base_all'; % 'AntonNormalize', 'Total_Crude', 'Clean_Crude', 'Total_Clean', 'P3quant_NSAF', 'P3quant_NadjSPC', 'Tip_vs_Base_NSAF', 'Tip_vs_Base_NadjSPC', 'P3_NSAF', 'P3_NadjSPC', 'Mezo_undiluted', 'Mezo_x5-diluted', 'L100519_GLEE', 'L100519_GLEE_norm', 'Tip_vs_Base_all_crap'
% [N,T]=xlsread(strcat(NAME,'.xlsx'));
% P=T(2:end,1); % protein names
% % for i=1:length(P)
% %     if isempty(P{i}), P{i}=num2str(N(i,1)); end
% % end
% % for i=1:length(P)
% %     if strcmp(P{i},'NaN'), error('NaN found at i=%d ...\n',i), end
% %     if isempty(P{i}), error('empty protein-name found at i=%d ...\n',i), end
% % end
% S={[1 2 3], [4 5 6]}; % give column-indices of the samples in N
% p=20; % number of bins
% fbL=1; % merge the lowest-signal bins so that the lowest bin contains atleast this percentage of proteins
% fbH=1; % merge the highest-signal bins so that the highest bin contains atleast this percentage of proteins
% binchoice='adaptive-log'; % 'equal' or 'adaptive-log' ('adaptive-raw' real bad!)
% fit_type='cubic'; % 'linear' or 'cubic'
% q=0.5; % quantile to use to select values to fit a polynomial through
% iter=1000; % number of resampling iterations
% fdr_level=0.05; % fdr level
% % ----------------------

% fprintf('\n---- STARTING ----\n');

% Check that all the supplied options are acceptable and assign them to
% variables
optfile=input('Enter name of file that contains the specified options: ','s');
if exist(optfile,'file')~=2, error('Cannot find file %s\n',optfile), end

fid=fopen(optfile);
if ~fid, error('Cannot open file %s\n',optfile), end
while 1
    tline=fgetl(fid);
    if ~ischar(tline), break, end
    r=regexp(tline,'(\S+)\s*\=\s*(\S+)','tokens');
    if isempty(r), continue, end
    if length(r)~=1, error('incorrect return from regexp\n'), end
    if length(r{1})~=2, error('incorrect number of tokens returned from regexp\n'), end
    option=lower(r{1}{1}); value=r{1}{2};
    switch option
        case 'filename'
            if exist(value,'file')~=2, error('Cannot find file %s\n',value)
            else
                try
                    [N,T]=xlsread(value);
                catch exception
                    % throw(exception);
                    fprintf('%s\n',exception.message);
                end
            end
            if size(N,1)==(size(T,1)-1), P=T(2:end,1);
            elseif size(N,1)==size(T,1), P=T(:,1);
            else error('please check that the spreadsheet is correctly formatted\n');
            end
        case 'num_replicates_a'
            % integer?
            if regexp(value,'\.'), error('specify an integer value for %s\n',option), end
            nA=str2num(value); S{1}=1:1:nA;
        case 'num_replicates_b'
            % integer?
            if regexp(value,'\.'), error('specify an integer value for %s\n',option), end
            nB=str2num(value); S{2}=length(S{1})+[1:1:nB];
        case 'num_bins'
            % integer?
            if regexp(value,'\.'), error('specify an integer value for %s, recommended value = 20\n',option), end
            p=str2num(value);
            if ((p<10)||(p>100)), error('specify a value between 10 and 100 for %s, recommended value = 20\n',option), end
        case 'merge_low'
            % float?
            % if isempty(regexp(value,'\.')), error('specify a floating-point value for %s\n',option), end
            fbL=str2num(value);
            if ((fbL<0)||(fbL>10)), error('specify a value between 0 and 10 for %s, recommended value = 1\n',option), end
        case 'merge_high'
            % float?
            % if isempty(regexp(value,'\.')), error('specify a floating-point value for %s\n',option), end
            fbH=str2num(value);
            if ((fbH<0)||(fbH>10)), error('specify a value between 0 and 10 for %s, recommended value = 1\n',option), end
        case 'binchoice'
            % string (adaptive, linear)?
            binchoice=lower(value);
            if ~(strcmp(binchoice,'adaptive') || strcmp(binchoice,'equal')), error('specify either adaptive or equal for %s',option), end
        case 'fit_type'
            % string (linear, cubic)?
            fit_type=lower(value);
            if ~(strcmp(fit_type,'cubic') || strcmp(fit_type,'linear')), error('specify either cubic or linear for %s',option), end
        case 'fit_quantile'
            % float in (0,1)?
            if isempty(regexp(value,'\.')), error('specify a floating-point value for %s, recommended value = 0.5\n',option), end
            q=str2num(value);
            if ((q<=0)||(q>=1)), error('specify a value between 0 and 1 for %s, recommended value = 0.5\n',option), end
        case 'num_iterations'
            % integer in [100 10000]?
            if regexp(value,'\.'), error('specify an integer value for %s, recommended value = 1000\n',option), end
            iter=str2num(value);
            if ((iter<100)||(iter>10000)), error('specify a value between 100 and 10000 for %s, recommended value = 1000\n',option), end
        case 'fdr_level'
            % float in (0,1)?
            if isempty(regexp(value,'\.')), error('specify a floating-point value for %s, recommended value = 0.05\n',option), end
            fdr_level=str2num(value);
            if ((fdr_level<=0)||(fdr_level>=1)), error('specify a value between 0 and 1 for %s, recommended value = 0.05\n',option), end
        case 'output_id'
            % string?
            out_id=value;
        otherwise
            error('option %s not understood!\n',option);
    end
end
fclose(fid);

% odir='out';
% [status, result] = system(strcat('md ',odir));
% if status~=0, error('Could not create output directory'); end
%
% ofile=strcat(odir,'/','temp.txt');
% fo=fopen(ofile,'w');
% fprintf(fo,'OUTPUT\n');
% fclose(fo);

fprintf('\n---- DONE READING OPTIONS ----\n');

% Print out the options so the user knows what is being used
% fprintf('');

% Prevent figures from popping up
set(0,'DefaultFigureVisible','off')
conditions={'A','B'};

% START THE ANALYSIS
xbar=[]; stdev=[];
for j=1:length(S)
    xbar=[xbar, mean(N(:,S{j}),2)];
    stdev=[stdev, std(N(:,S{j}),0,2)];
end

OFILE=strcat(out_id,'.selected_points.txt');
fout=fopen(OFILE,'w');
fprintf(fout,'log(xbar)\tlog(stdev)\n');
if strcmp(binchoice,'equal')
    % ---- EQUAL SIZED BINS ----
    adjRsq=[]; C={};
    for j=1:length(S)
        fprintf('-- processing condition %s --\n',conditions{j});
        X=[]; Y=[];
        F=find(xbar(:,j)>0); xbar_values=log(xbar(F,j)); stdev_values=log(stdev(F,j));        
        if length(find(isfinite(stdev_values)))~=length(stdev_values) %, error('stdev_values contains zero values!'), end
            F=find(isfinite(stdev_values)); xbar_values=xbar_values(F); stdev_values=stdev_values(F);
        end
        if length(find(isfinite(stdev_values)))~=length(stdev_values), error('stdev_values contains zero values!'), end
        [sorted_xbar_values,I]=sort(xbar_values); sorted_stdev_values=stdev_values(I);                
        L=floor(length(sorted_xbar_values)/p);                
        % divide into p bins of equal size, take median
        for i=1:p-1
            X(i,1)=quantile(sorted_xbar_values((i-1)*L+1:i*L),q);
            Y(i,1)=quantile(sorted_stdev_values((i-1)*L+1:i*L),q);
        end
        X(p,1)=quantile(sorted_xbar_values((p-1)*L+1:end),q);
        Y(p,1)=quantile(sorted_stdev_values((p-1)*L+1:end),q);
        IX=isfinite(X); IY=isfinite(Y);
        if length(find(IX==IY))~=length(IX), error('IX and IY are not identical'), end
        X=X(IX); Y=Y(IY); % remove NaNs, if any
        % print out selected points
        fprintf(fout,'---- condition %s ----\n',conditions{j});
        for i=1:length(X), fprintf(fout,'%f\t%f\n',X(i),Y(i)); end
        % perform the fit
        if strcmp(fit_type,'linear')
            [cfun,gof,output] = fit(X,Y,'poly1');
            fprintf('adjRsq = %f\n',gof.adjrsquare);
            adjRsq=[adjRsq, gof.adjrsquare];
            C{j}=[cfun.p1; cfun.p2];
            Yhat=[X ones(length(X),1)]*C{j};
        elseif strcmp(fit_type,'cubic')
            [cfun,gof,output] = fit(X,Y,'poly3');
            fprintf('adjRsq = %f\n',gof.adjrsquare);
            adjRsq=[adjRsq, gof.adjrsquare];
            C{j}=[cfun.p1; cfun.p2; cfun.p3; cfun.p4];
            Yhat=[X.^3 X.^2 X ones(length(X),1)]*C{j};
        else
            error('Could not understand fit_type\n');
        end
        fitdata=[X Y Yhat];
        % save(strcat(NAME,'.',strcat(int2str(p),'_',binchoice),'.',fit_type,'.',strcat('SAMPLE-',int2str(j)),'.fitdata.mat'),'fitdata');        
        % leave out the outliers from the plot
        showpoints=find((stdev_values>(mean(stdev_values)-5*std(stdev_values)))&(stdev_values<(mean(stdev_values)+5*std(stdev_values))));
        figure, plot(xbar_values(showpoints),stdev_values(showpoints),'b.',X,Yhat,'r-'), legend('raw',strcat('fit (adjRsq=',num2str(adjRsq(j)),')'))
        xlabel('log(xbar)'), ylabel('log(stdev)'), title(strcat('SAMPLE-',int2str(j)));
        saveas(gcf,strcat(out_id,'.sample-',int2str(j),'.jpg'),'jpg');
        figure, plot(1:length(sorted_xbar_values),sorted_xbar_values,'b.')
        xlabel('protein #'), ylabel('signal level (xbar)'), title(strcat('SAMPLE-',int2str(j)));
        saveas(gcf,strcat(out_id,'.sample-',int2str(j),'.siglevel.jpg'),'jpg');
    end
elseif strcmp(binchoice,'adaptive')
    % ---- ADAPTIVELY SIZED BINS (LOG SIGNAL) ----
    adjRsq=[]; C={};
    for j=1:length(S)
        fprintf('-- processing condition %s --\n',conditions{j});
        X=[]; Y=[];
        F=find(xbar(:,j)>0); xbar_values=log(xbar(F,j)); stdev_values=log(stdev(F,j)); % xbar_values=xbar(F,j); stdev_values=stdev(F,j);
        if length(find(isfinite(stdev_values)))~=length(stdev_values) %, error('stdev_values contains zero values!'), end
            F=find(isfinite(stdev_values)); xbar_values=xbar_values(F); stdev_values=stdev_values(F);
        end
        if length(find(isfinite(stdev_values)))~=length(stdev_values), error('stdev_values contains zero values!'), end
        [sorted_xbar_values,I]=sort(xbar_values); sorted_stdev_values=stdev_values(I);
        % divide the range of signal into p equal portions
        dx=(sorted_xbar_values(end)-sorted_xbar_values(1))/p;
        Start=[]; Stop=[];
        for i=1:p-1
            if isempty(Start), start=1; else start=Stop(end)+1; end
            stop=max(find(sorted_xbar_values<=(sorted_xbar_values(1)+i*dx)));
            % fprintf('\ni=%d, start=%d, stop=%d\n',i,start,stop);
            if stop>=start, Start=[Start; start]; Stop=[Stop; stop]; end
        end
        start=Stop(end)+1; stop=length(sorted_xbar_values);
        if stop>=start, Start=[Start; start]; Stop=[Stop; stop]; end
        % disp([Start Stop])
        if length(Start)~=length(Stop), error('Start and Stop are not of same length!\n'); end
        % merge the low-signal bins so that the lowest one contains atleast
        % fbL of the proteins
        ind=min(find(Stop>((fbL/100)*length(F))));
        Start=[1; Start(ind+1:end)]; Stop=Stop(ind:end);
        fprintf('Percentage of proteins in the lowest bin = %f\n',100*(Stop(1)-Start(1)+1)/length(F));
        % merge the high-signal bins so that the highest one contains atleast
        % fbH of the proteins
        ind=max(find(Stop<((1-(fbH/100))*length(F))));
        Start=Start(1:(ind+1)); Stop=[Stop(1:ind); Stop(end)];
        fprintf('Percentage of proteins in the highest bin = %f\n',100*(Stop(end)-Start(end)+1)/length(F));
        % disp([Start Stop])
        % extract the quantiles in each bin
        for i=1:length(Start)
            X(i,1)=quantile(sorted_xbar_values(Start(i):Stop(i)),q);
            Y(i,1)=quantile(sorted_stdev_values(Start(i):Stop(i)),q);
        end
        IX=isfinite(X); IY=isfinite(Y);
        if length(find(IX==IY))~=length(IX), error('IX and IY are not identical'), end
        X=X(IX); Y=Y(IY); % remove NaNs, if any        
        % print out selected points
        fprintf(fout,'---- condition %s ----\n',conditions{j});
        for i=1:length(X), fprintf(fout,'%f\t%f\n',X(i),Y(i)); end
        % perform the fit        
        if strcmp(fit_type,'linear')
            [cfun,gof,output] = fit(X,Y,'poly1');
            fprintf('adjRsq = %f\n',gof.adjrsquare);
            adjRsq=[adjRsq, gof.adjrsquare];
            C{j}=[cfun.p1; cfun.p2];
            Yhat=[X ones(length(X),1)]*C{j};
        elseif strcmp(fit_type,'cubic')
            [cfun,gof,output] = fit(X,Y,'poly3');
            fprintf('adjRsq = %f\n',gof.adjrsquare);
            adjRsq=[adjRsq, gof.adjrsquare];
            C{j}=[cfun.p1; cfun.p2; cfun.p3; cfun.p4];
            Yhat=[X.^3 X.^2 X ones(length(X),1)]*C{j};
        else
            error('Could not understand fit_type\n');
        end
        fitdata=[X Y Yhat];
        % save(strcat(NAME,'.',strcat(int2str(p),'_',binchoice),'.',fit_type,'.',strcat('SAMPLE-',int2str(j)),'.fitdata.mat'),'fitdata');
        % leave out the outliers from the plot
        showpoints=find((stdev_values>(mean(stdev_values)-5*std(stdev_values)))&(stdev_values<(mean(stdev_values)+5*std(stdev_values))));
        figure, plot(xbar_values(showpoints),stdev_values(showpoints),'b.',X,Yhat,'r-'), legend('raw',strcat('fit (adjRsq=',num2str(adjRsq(j)),')'))
        xlabel('log(xbar)'), ylabel('log(stdev)'), title(strcat('SAMPLE-',int2str(j)));
        saveas(gcf,strcat(out_id,'.sample-',int2str(j),'.jpg'),'jpg');
        % figure, plot(xbar_values,stdev_values,'b.',X,Y,'rs',X,Yhat,'r-'), legend('raw','selected',strcat('fit (adjRsq=',num2str(adjRsq(j)),')'))
        figure, plot(1:length(sorted_xbar_values),sorted_xbar_values,'b.')
        xlabel('protein #'), ylabel('signal level (xbar)'), title(strcat('SAMPLE-',int2str(j)));
        saveas(gcf,strcat(out_id,'.sample-',int2str(j),'.siglevel.jpg'),'jpg');
    end
end
fclose(fout);


% choose the fit that gives better adjRsq
bestfitind=find(adjRsq==max(adjRsq));
if length(bestfitind)~=1, error('trouble finding the fit with best adjRsq'), end
coeff=C{bestfitind};

% replace xbar zeroes with min-positive xbar from same sample
for j=1:length(S)
    xbar(find(xbar(:,j)<=0),j)=min(xbar(find(xbar(:,j)>0),j));
end
min_xbar_value=min(xbar);

% calculated model-based STN
model_stn=(xbar(:,2)-xbar(:,1))./(calcs(xbar(:,1),coeff)+calcs(xbar(:,2),coeff));
if length(find(isfinite(model_stn)))~=length(model_stn), error('model_stn contains not all finite values...\n'), end

% calculate null distribution of model_stn using the specified baseline condition
baselinecol=bestfitind; % sample that is to be used as baseline
fprintf('\nUsing condition %s as baseline, since its adjRsq is higher\n',conditions{baselinecol});
othercol=nan; % the other sample
if baselinecol==1, othercol=2; else othercol=1; end
% notation followed: A=baseline, B=other
nA=length(S{baselinecol}); nB=length(S{othercol});
model_stn_dist=nan(iter*size(N,1),1);
fprintf('-- doing re-sampling --\n');
for i=1:size(N,1)
    if rem(i,100)==0, fprintf('processing %d of %d ...\n',i,size(N,1)), end
    A=N(i,S{baselinecol}); % B=N(i,S{othercol});
    % sample values with replacement
    Astar=A(randi(nA,[iter,nA])); Bstar=A(randi(nB,[iter,nB]));
    xbar_Astar=mean(Astar,2); xbar_Bstar=mean(Bstar,2);
    IA=find(xbar_Astar==0); xbar_Astar(IA)=min_xbar_value(baselinecol);
    IB=find(xbar_Bstar==0); xbar_Bstar(IB)=min_xbar_value(baselinecol);
    this_dist=(xbar_Bstar-xbar_Astar)./(calcs(xbar_Astar,coeff)+calcs(xbar_Bstar,coeff));
    if length(find(isfinite(this_dist)))~=iter, error('check resampled distribution for i=%d\n',i), end
    model_stn_dist(iter*(i-1)+1:i*iter)=this_dist;
end
figure, hist(model_stn_dist,100), title('Histogram of stn distribution')
saveas(gcf,strcat(out_id,'.stn_distr.jpg'),'jpg');

% calculate p-values
pValue=nan(size(N,1),1);
model_stn_dist=sort(model_stn_dist);
for i=1:size(N,1)
    % -- SLOW --
    % c=length(find(model_stn_dist<=model_stn(i)))/length(model_stn_dist); pValue(i)=2*min(c,1-c);
    % -- FAST --
    I=find(model_stn_dist>model_stn(i),1); if isempty(I), pValue(i)=0; else c=(I-1)/length(model_stn_dist); pValue(i)=2*min(c,1-c); end
end

% plot STN vs p-values
figure, plot(model_stn,pValue,'r*'), xlabel('STN'), ylabel('p-value');
% save(strcat(NAME,'.',strcat(int2str(p),'_',binchoice),'.',fit_type,'.stn.mat'),'model_stn'); save(strcat(NAME,'.',strcat(int2str(p),'_',binchoice),'.',fit_type,'.pValue.mat'),'pValue');
saveas(gcf,strcat(out_id,'.stn_pvalue.jpg'),'jpg');

% apply fdr cutoff
H=fdr(pValue,fdr_level);
fprintf('number of DEG detected = %d\n',length(find(H)));
[pValue,order]=sort(pValue); H=H(order); P=P(order); N=N(order,:); model_stn=model_stn(order); xbar=xbar(order,:);

% close all figures
close all

% print out DEG
OFILE=strcat(out_id,'.DEG.txt');
fout=fopen(OFILE,'w');
fprintf(fout,'Protein\t');
for j=1:nA, fprintf(fout,'A_%d\t',j); end
for j=1:nB, fprintf(fout,'B_%d\t',j); end
fprintf(fout,'xbar(A)\ts(A)\txbar(B)\ts(B)\tSTN\tpValue\tSignificantDiffExp\n');
for i=1:length(H)
    fprintf(fout,'%s',P{i});
    for j=1:size(N,2), fprintf(fout,'\t%6.12f',N(i,j)); end
    fprintf(fout,'\t%6.12f\t%6.12f\t%6.12f\t%6.12f\t%6.12f\t%6.12f',xbar(i,1),calcs(xbar(i,1),coeff),xbar(i,2),calcs(xbar(i,2),coeff),model_stn(i),pValue(i));
    if H(i)==1, fprintf(fout,'\tYes\n'); else fprintf(fout,'\tNo\n'); end
end
fclose('all');
