clear;
%
% Parameters.
%
tr_seed = 54316377; te_seed = 41606561; sg_seed = 24010204;         % Seeds.
tr_p = 20000; te_q = tr_p/10; tr_freq = 0.0;                        % Datasets generation.
epsG = 10^-6; kmax = 1000;                                    % Stopping condition.
ils=3; ialmax = 1; kmaxBLS=30; epsal=10^-3; c1=0.01; c2=0.45; % Linesearch.
icg = 0; irc = 0; nu = 0.0;                                   % Search direction.
sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;                        % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.
%
% Optimization
%
global iheader; iheader = 1;
csvfile = strcat('uo_nn_batch_',num2str(tr_seed),'-',num2str(te_seed),'-',num2str(sg_seed),'.csv');
fileID = fopen(csvfile ,'w');
t1=clock;
for num_target = [1:10]
        for isd = [1,3,7]
            if isd == 1
                la = 0.01;
            elseif isd ==3
                la = 0.01;
            elseif isd == 5
                la = 0.0; % aquests valors són pels resultats obtinguts amb les seeds anteriors.
            end

            [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
            if iheader == 1
                fprintf(fileID,'num_target;      la; isd;  niter;     tex; tr_acc; te_acc;        L*;\n');
            end
            fprintf(fileID,'         %1i; %7.4f;   %1i; %6i; %7.4f;  %5.1f;  %5.1f;  %8.2e;\n', mod(num_target,10), la, isd, niter, tex, tr_acc, te_acc, fo);
            iheader=0;
        end
end
t2=clock;
total_t = etime(t2,t1);
fprintf(' wall time = %6.1d s.\n', total_t);
fclose(fileID);
