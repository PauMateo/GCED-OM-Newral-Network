%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)

% Input parameters:
%
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SGÇ_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
%
% Output parameters:
%
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     fo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%
t1=clock;
Xtr= 0; ytr= 0; wo= 0; fo= 0; tr_acc= 0; Xte= 0; yte= 0; te_acc= 0; niter= 0; tex= 0;
te_freq=0.0;

fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n')
fprintf('[uo_nn_solve] %s\n',datetime)
fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate TRAINING data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[uo_nn_solve] Training data set generation.\n')

[Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
%uo_nn_Xyplot(Xtr,ytr,[]);

fprintf('[uo_nn_solve]      num_target = %d ', num_target);
fprintf('\n');
fprintf('[uo_nn_solve]      tr_freq = %3.2f \n', tr_freq);
fprintf('[uo_nn_solve]      tr_p = %d \n', tr_p);
fprintf('[uo_nn_solve]      tr_seed = %d \n', tr_seed)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate TEST data set
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[uo_nn_solve] Test data set generation.\n');

[Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);
%uo_nn_Xyplot(Xte,yte,[]);

fprintf('[uo_nn_solve]      te_freq = %3.2f \n', te_freq);
fprintf('[uo_nn_solve]      te_p = %d \n', te_q);
fprintf('[uo_nn_solve]      te_seed = %d \n', te_seed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[uo_nn_solve] Optimization\n');
sig = @(Xtr) 1./(1+exp(-Xtr));
y = @(Xtr,w) sig(w'*sig(Xtr));
L = @(w) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+ (la*norm(w)^2)/2;
gL = @(w) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;
L_tilla = @(w,Xtr,ytr,la) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+ (la*norm(w)^2)/2;
gL_tilla = @(w,Xtr,ytr,la) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;

wo = zeros(35,1);
if isd == 7
    [wk,dk,alk,wo] = uo_SGM(wo,la,L_tilla,gL_tilla,Xtr,ytr,Xte,yte,sg_seed, sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
else
    [wk,dk,alk,iWk,Hk] = uo_solve_proj(wo,L,gL,epsG,kmax,ialmax,c1,c2,ils,isd,icg,irc,nu,Xtr,ytr,kmaxBLS,epsal);
    niter = size(wk);
    niter = niter(2);

    wo = wk(:,niter-1);
end

fprintf('[uo_nn_solve]      L2 reg. lambda = %+3.4f \n', la);
fprintf('[uo_nn_solve]      epsG = %3.1e, kmax = %d \n', epsG,kmax);
fprintf('[uo_nn_solve]      ils = %d, almax = %d kmaxBLS = %d epsBLS = %+3.1e \n', ils,ialmax,kmaxBLS,epsal);
fprintf('[uo_nn_solve]      c1 = %+3.2e, c2 = %+3.2e  isd = %d  \n', c1,c2,isd);
fprintf('[uo_nn_solve]      w0 = [0]');
fprintf('\n');

if isd ==1
    fprintf('[uo_nn_solve]      k     al   iW       g''*d    f       ||g||    \n');
elseif isd == 3
    fprintf('[uo_nn_solve]      k     g''*d        al iW    la(1)    kappa    ||g||        f         r        M\n');   
elseif isd ==7
    fprintf('[uo_nn_solve]      k     al    iW    g''*d    f       ||g||    \n');

end
niter = size(wk);
niter = niter(2);
for k = 1:3
    if isd == 1 
   
    fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.2e %+3.2e \n', k, alk(k), iWk(k), gL(wk(:,k))'*dk(:,k), L(wk(:,k)), norm(gL(wk(:,k))));
    elseif isd == 3
   fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.1e %+3.1e %+3.1e %+3.1e \n', k, gL(wk(:,k))'*dk(:,k), alk(k), iWk(k), norm(gL(wk(:,k))), L(wk(:,k)));
    
    elseif isd == 7
        fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.2e %+3.2e \n', k, alk(k), 0, gL(wk(:,k))'*dk(:,k), L(wk(:,k)), norm(gL(wk(:,k))));
    end
end
for k = niter-4:niter-1
    if isd == 1 
   
    fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.2e %+3.2e \n', k, alk(k), iWk(k), gL(wk(:,k))'*dk(:,k), L(wk(:,k)), norm(gL(wk(:,k))));
    elseif isd == 3
    fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.2e %+3.2e \n', k, alk(k), iWk(k), gL(wk(:,k))'*dk(:,k), L(wk(:,k)), norm(gL(wk(:,k))));
     elseif isd == 7
        fprintf('[uo_nn_solve]      %6d %+3.1e %d %3.2e %+3.2e %+3.2e \n', k, alk(k), 0, gL(wk(:,k))'*dk(:,k), L(wk(:,k)), norm(gL(wk(:,k))));
    
    end
end

if isd ==1
fprintf('[uo_nn_solve]      k     al   iW       g''*d    f       ||g||    \n');
elseif isd == 3
fprintf('[uo_nn_solve]      k     g''*d        al iW    la(1)    kappa    ||g||        f         r        M\n');   
elseif isd ==7
    fprintf('[uo_nn_solve]      k     al    iW    g''*d    f       ||g||    \n');

end
fo = L(wo);
fprintf('[uo_nn_solve]      wo=[\n');
for i = 1:7
    fprintf('[uo_nn_solve]              ')
    for j = 1:5
        fprintf('%3.1e,',wo(i+j,:));
    end
    fprintf('\n');
end
fprintf('[uo_nn_solve]      ] \n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[uo_nn_solve] Accuracy.\n');

vec = [];
for i = 1:tr_p
    vec = [vec, (round(y(Xtr(:,i), wo)) == ytr(i))];
end

tr_acc = (100/tr_p) * sum(vec);
fprintf('[uo_nn_solve] tr_accuracy = %+3.1f\n', tr_acc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vec = [];
for i = 1:te_q
    vec = [vec, (round(y(Xte(:,i), wo)) == yte(i))];
end

te_acc = (100/te_q) * sum(vec);

fprintf('[uo_nn_solve] te_accuracy = %+3.2f\n', te_acc);

t2 = clock;
tex = etime(t2,t1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
