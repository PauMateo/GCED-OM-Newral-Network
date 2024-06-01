function [wk,dk,alk,wo] = uo_SGM(w, la, L_tilla, gL_tilla, Xtr,ytr,Xte,yte,sg_seed, sg_al0,sg_be ,sg_ga,sg_emax, sg_ebest)
    mida_p = size(Xtr,2);
    vector_p = 1:mida_p;
    m = floor(sg_ga*mida_p); sg_ke = ceil(mida_p/m); sg_kmax = sg_ke*sg_emax; e=0; s=0; Lte_best = inf; k=0;
    wk = [w]; dk = []; alk = []; rng(sg_seed);
    disp(ceil(mida_p/m - 1));
    while e <= sg_emax && s < sg_ebest
        P = vector_p(randperm(mida_p));

        for i = 0:ceil(mida_p/m - 1)
            
            S = P(i*m+1:min((i+1)*m,mida_p));
            Xtrs = []; ytrs = [];
            for i = 1:m
                Xtrs = [Xtrs,Xtr(:,S(i))];
                ytrs = [ytrs,ytr(S(i))];
            end
            d = -gL_tilla(w, Xtrs,ytrs,la);
            if k <= floor(sg_kmax*sg_be)
                al = (1-k/(floor(sg_kmax*sg_be)))*sg_al0 + k/floor(sg_kmax*sg_be) * 0.01 * sg_al0;
            else
                al = 0.01 * sg_al0;
            end

            w = w + al * d; k = k + 1;
            wk = [wk, w]; dk = [dk, d]; alk = [alk,al];
                           
        end
        e = e + 1;
        Lte = L_tilla(w, Xte, yte, la);
        if Lte < Lte_best
            Lte_best = Lte; wo = w; s = 0;
        else
            s = s+ 1;
        end
    end
end
