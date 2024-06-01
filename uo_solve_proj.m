function [wk,dk,alk,iWk,Hk] = uo_solve_proj(w,L,gL,epsG,kmax,ialmax,c1,c2,ils,isd,icg,irc,nu,Xtr,ytr,kmaxBLS,epsal)

n=size(w,1);

wk = []; dk = []; alk = []; iWk = []; Hk=zeros(n);
I = eye(n,n);
H = I;
k = 1;
ldescent = true;
almax=1;

while norm(gL(w)) > epsG & k < kmax & ldescent  %............... main loop
    
    if isd == 1 % GM
        d = -gL(w);   

    elseif isd == 3 % BFGS
        d = - H * gL(w);
        Hk=[Hk,H];
    end
    
    ldescent = d'*gL(w) < 0;

    if ils == 1
        h = jacobian(gL);
        Q = h(zeros(n,1));
        alphas = -(gL(w)'*d)/(d'*Q*d);
        iout=3;
    
    elseif ils == 2      % BLS, constant reduction.
        almin = 0.001;
        rho = 0.5;
        [alphas, iout] = uo_BLS(w,d,L,gL,almax,almin,rho,c1,c2,iW);
    elseif ils == 3
        % set almax for iteration k:
        if k~=1
            if ialmax == 1
                almax = 2*(L(w) - L(w_ant))/(gL(w)'*d);
            elseif ialmax == 2
                almax = alk(k-1) * (gL(w_ant)'*d_ant) / (gL(w)'*d);
            end
        end

        [alphas,iout] = uo_BLSNW32(L,gL,w,d,almax,c1,c2,kmaxBLS,epsal);
        
    end

     % Update %%%%%%%%%%%%%%%%%%%
    w_ant = w; d_ant = d;
    w = w + alphas*d; k=k+1;

    wk = [wk,w];
    dk = [dk,d];
    alk = [alk,alphas];
    iWk = [iWk,iout];
    if isd == 3
        s = w - w_ant;
        y = gL(w) - gL(w_ant);
        p = 1/(y'*s);
        H = (I - p*s*y')*H*(I - p*y*s') + p*(s*s') ;
    end
end


end
